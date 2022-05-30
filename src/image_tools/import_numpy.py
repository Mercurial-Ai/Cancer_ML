from xml.sax.handler import feature_namespaces
import numpy as np
from math import sqrt
import tensorflow as tf
from stl import mesh
import trimesh
from trimesh import Trimesh
from trimesh import voxel
from trimesh.voxel import creation
import os
import matplotlib.pyplot as plt

from src.image_tools.filter_ids import filter_ids
from src.image_tools.remove_ids import remove_ids
from dcm_npy_loader.main import dcm_npy_loader

def import_numpy_2d(path, clinical_ids, crop_size=(512, 512)):
    patients, ids, sliceLocs = dcm_npy_loader(path)

    i = 0
    for p in patients:
        p_id = ids[i]
        if not p_id in clinical_ids:
            del patients[i]
        i = i + 1

    slice_locations_min = []
    slice_locations_max = []
    for sl in sliceLocs:
        mi = np.amin(sl)
        ma = np.amax(sl)
        slice_locations_min.append(mi)
        slice_locations_max.append(ma)

    # subinterval length (mm)
    subinterval_length = 7
    interval_nums = []
    all_intervals = []
    for ma, mi in zip(slice_locations_max, slice_locations_min):
        interval_num = int(round((ma - mi)/subinterval_length, 0))
        interval_nums.append(interval_num)

        interval_num = min(interval_nums)

        # set min slice loc as initial interval marker
        intervals = []
        interval_marker = mi
        for i in range(interval_num):
            intervals.append(interval_marker)
            interval_marker = interval_marker+subinterval_length*i

        all_intervals.append(intervals)

    # array to store all intervals; dim 1 is set to length of dataset
    all_interval_imgs = np.empty(shape=(len(clinical_ids), len(intervals), 512**2+2), dtype=np.float16)

    p = 0
    for p_id in ids:
        for i in range(len(intervals)):
            if i < (len(intervals)-1):
                low = intervals[i]
                high = intervals[i+1]

                interval_imgs = np.empty(shape=(len(intervals), 512, 512), dtype=np.float16)

                j = 0
                for p in patients:
                    id = ids[j]

                    if int(id) == int(p_id):

                        k = 0
                        for s in p:
                            slice_location=sliceLocs[j][k]

                            if slice_location < high and slice_location > low:
            
                                interval_imgs[i] = s

                                # break after adding one image from the interval
                                break

                            k = k + 1
                    
                    j = j + 1

                all_interval_imgs[p] = interval_imgs

        p = p + 1

    print(all_interval_imgs)
    print(all_interval_imgs.shape)

    return all_interval_imgs

def import_numpy_3d(dir):
    patients = os.listdir(dir)
    all_vertices = []
    for patient in patients:
        patient = os.path.join(dir, patient)
        patient_mesh = mesh.Mesh.from_file(patient)
        vertices = np.around(np.unique(patient_mesh.vectors.reshape(int(patient_mesh.vectors.size/3), 3), axis=0),2)
        vertices = np.expand_dims(vertices, 0)
        all_vertices.append(vertices)

    vertice_nums=[]
    for vertice in all_vertices:
        vertice_num = vertice.shape[1]
        vertice_nums.append(vertice_num)

    smallest_patient_size = min(vertice_nums)

    i = 0
    for vertice in all_vertices:
        vertice = tf.image.random_crop(vertice, (1, smallest_patient_size, 3))
        all_vertices[i] = np.asarray(vertice)
        i = i + 1

    vertices = np.concatenate(all_vertices, axis=0)

    return vertices

import_numpy_3d("C:\\Users\\trist\\cs_projects\\Cancer_Project\\Cancer Imagery\\3d_duke")
