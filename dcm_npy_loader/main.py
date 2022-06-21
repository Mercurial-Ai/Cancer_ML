import pickle
import os
import pydicom
from torch.utils.data import Dataset
import numpy as np

class dcm_npy_loader(Dataset):

    def __init__(self, img_dir, load, shape=(512, 512)):
        self.img_dir = img_dir
        self.load = load
        self.shape = shape

        if not self.load:
            load_paths = list()
            for (dirpath, dirnames, filenames) in os.walk(img_dir):
                load_paths += [os.path.join(dirpath, file) for file in filenames]

            files = []
            ids = []
            for path in load_paths:
                try:
                    file = pydicom.dcmread(path)
                    if file.pixel_array.shape == self.shape:
                        files.append(file)
                        id = file.PatientID
                        ids.append(id)
                except:
                    print("Image " + path + " could not be loaded")

            # remove duplicates
            ids = list(set(ids))

            all_img3d = []
            all_ids = []
            all_sliceLocs = []
            for id in ids:

                # skip files with no SliceLocation
                slices = []
                skipcount = 0
                for f in files:
                    if f.PatientID == id:
                        if hasattr(f, 'SliceLocation'):
                            slices.append(f)
                        else:
                            skipcount = skipcount + 1

                # ensure slices are in the correct order
                slices = sorted(slices, key=lambda s: s.SliceLocation)

                # pixel aspects, assuming all slices are the same
                ps = slices[0].PixelSpacing
                ss = slices[0].SliceThickness
                ax_aspect = ps[1]/ps[0]
                sag_aspect = ps[1]/ss
                cor_aspect = ss/ps[0]

                # create 3D array
                img_shape = list(slices[0].pixel_array.shape)
                img_shape.append(len(slices))
                img3d = np.zeros(img_shape, dtype=np.int8)

                p_id = slices[0].PatientID
                # get only numbers from patient id
                p_id = [int(s) for s in p_id if s.isdigit()]
                p_id = int(''.join([str(i) for i in p_id]))

                slice_locs = []

                # fill 3D array with the images from the files
                for i, s in enumerate(slices):
                    img2d = s.pixel_array

                    slice_locs.append(s.get('SliceLocation'))
                    if list(img2d.shape) == img_shape[:2]:
                        img3d[:, :, i] = img2d

                all_ids.append(p_id)
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
