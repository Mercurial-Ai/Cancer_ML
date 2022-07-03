import pickle
import os
import pydicom
import torchio
from torch.utils.data import Dataset
import numpy as np
import torch
import time

class dcm_npy_loader(Dataset):

    def __init__(self, img_dir, load, shape=(512, 512)):
        self.img_dir = img_dir
        self.load = load
        self.shape = shape

        if not self.load:
            load_paths = list()
            for (dirpath, dirnames, filenames) in os.walk(img_dir):
                load_paths += [os.path.join(dirpath, file) for file in filenames]

            ids = np.array([], dtype=np.int8)
            img_list = []
            num_exceptions = 0
            for path in load_paths:
                try:
                    file = pydicom.dcmread(path)
                    if file.pixel_array.shape == self.shape:
                        id = file.PatientID
                        for c in id:
                            if not c.isdigit():
                                id = id.replace(c, '')

                        ids = np.append(ids, id)

                        subject_dict = {
                            'one image': torchio.ScalarImage(path),
                            'id':id,
                            'SliceLocation':file.SliceLocation
                        }
                        subject = torchio.Subject(subject_dict)

                        img_list.append(subject)
                except:
                    if num_exceptions < 5:
                        print("Image " + path + " could not be loaded")
                    elif num_exceptions == 5:
                        print("More than 5 exceptions occured")
                    
                    num_exceptions = num_exceptions + 1

            img_list.sort(key=lambda s: int(s['id']))

            ids = set([int(s['id']) for s in img_list])

            all_img3d = []
            all_ids = []
            all_sliceLocs = []
            for id in ids:
                prev_time_id = time.time()

                slices = []
                i = 0
                for slice in img_list:

                    p_id = int(slice['id'])

                    if int(p_id) == int(id):
                            
                        slices.append(slice)

                        # remove slices from dataset that have already been appended
                        del img_list[i]

                    i = i + 1

                id_slices = torchio.SubjectsDataset(slices)

                # ensure slices are in the correct order
                id_slices = sorted(id_slices, key=lambda s: s.SliceLocation)

                # create 3D array
                img_shape = list(id_slices[0]['one image']['data'].shape)
                img_shape.append(len(id_slices))
                img3d = np.empty(img_shape, dtype=np.int8)

                p_id = id_slices[0]['id']
                # get only numbers from patient id
                p_id = [int(s) for s in str(p_id) if s.isdigit()]
                p_id = int(''.join([str(i) for i in p_id]))

                slice_locs = []

                # fill 3D array with the images from the files
                for i, s in enumerate(id_slices):
                    img2d = s['one image']['data']

                    slice_locs.append(s['SliceLocation'])
                    if list(img2d.shape) == img_shape[:4]:
                        img3d[:, :, :, :, i] = img2d

                all_ids.append(p_id)
                img3d = np.squeeze(img3d)
                all_img3d.append(img3d)
                all_sliceLocs.append(slice_locs)

                aft_time_id = time.time()
                id_load_time = aft_time_id - prev_time_id

                print("id: " + str(id) + " has completed " + "in " + str(round(id_load_time,0)) + " seconds!")

            self.data = [all_img3d, all_ids, all_sliceLocs]

            with open('data/Duke-Breast-Cancer-MRI/data', 'wb') as fp:
                pickle.dump(self.data, fp)
        
        else:
            with open('data/Duke-Breast-Cancer-MRI/data', 'rb') as fp:
                self.data = pickle.load(fp)

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        sample = [self.data[0][idx], self.data[1][idx], self.data[2][idx]]
        return sample
