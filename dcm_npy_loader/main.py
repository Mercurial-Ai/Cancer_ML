import pickle
import os
import pydicom
import torchio
from torch.utils.data import Dataset
import numpy as np
import torch

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
                    print("Image " + path + " could not be loaded")

            slice_dataset = torchio.SubjectsDataset(img_list, load_getitem=True)

            # remove duplicates
            ids = list(set(ids))

            all_img3d = []
            all_ids = []
            all_sliceLocs = []
            for id in ids:

                slices = []
                for slice in slice_dataset:
        
                    p_id = int(slice['id'])
                    image = slice['one image']

                    if int(p_id) == int(id):
                        slice = torchio.Subject({
                            'one image': image,
                            'id': p_id,
                            'SliceLocation': slice['SliceLocation']})
                            
                        slices.append(slice)
        
                id_slices = torchio.SubjectsDataset(slices)

                # ensure slices are in the correct order
                id_slices = sorted(id_slices, key=lambda s: s.SliceLocation)

                # create 3D array
                img_shape = list(id_slices[0]['one image']['data'].shape)
                img_shape.append(len(id_slices))
                img3d = np.zeros(img_shape, dtype=np.int8)

                p_id = id_slices[0]['id']
                # get only numbers from patient id
                p_id = [int(s) for s in str(p_id) if s.isdigit()]
                p_id = int(''.join([str(i) for i in p_id]))

                slice_locs = []

                # fill 3D array with the images from the files
                for i, s in enumerate(id_slices):
                    img2d = s['one image']['data']

                    slice_locs.append(s['SliceLocation'])
                    if list(img2d.shape) == img_shape[:2]:
                        img3d[:, :, i] = img2d

                all_ids.append(p_id)
                img3d = np.squeeze(img3d)
                all_img3d.append(img3d)
                all_sliceLocs.append(slice_locs)

            self.data = [all_img3d, all_ids, all_sliceLocs]

            with open('data\\Duke-Breast-Cancer-MRI\\data', 'wb') as fp:
                pickle.dump(self.data, fp)
        
        else:
            with open('data\\Duke-Breast-Cancer-MRI\\data', 'rb') as fp:
                self.data = pickle.load(fp)

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        sample = [self.data[0][idx], self.data[1][idx], self.data[2][idx]]
        return sample
