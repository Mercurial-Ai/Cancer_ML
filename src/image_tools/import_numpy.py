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

def import_numpy_2d(path, clinical_ids, crop_size=(512, 512)):
    img_array = np.load(path)

    img_array = filter_ids(img_array, clinical_ids)

    slice_locations_min = np.amin(img_array[:, -2])
    slice_locations_max = np.amax(img_array[:, -2])

    # subinterval length (mm)
    subinterval_length = 7
    interval_num = int(round((slice_locations_max - slice_locations_min)/subinterval_length, 0))

    # set min slice loc as initial interval marker
    intervals = []
    interval_marker = slice_locations_min
    for i in range(interval_num):
        intervals.append(interval_marker)
        interval_marker = interval_marker+subinterval_length*i

    # array to store all intervals; dim 1 is set to length of dataset
    all_interval_imgs = np.empty(shape=(len(clinical_ids), len(intervals), 512**2+2), dtype=np.float16)

    p = 0
    for p_id in clinical_ids:
        for i in range(len(intervals)):
            if i < (len(intervals)-1):
                low = intervals[i]
                high = intervals[i+1]

                interval_imgs = np.empty(shape=(len(intervals), 512**2+2), dtype=np.float16)

                for image in img_array:
                    id = image[-1]
                    if int(id) == int(p_id):
                        slice_location=image[-2]
                        if slice_location < high and slice_location > low:
            
                            interval_imgs[i] = image

                            print(interval_imgs)

                            # break after adding one image from the interval
                            break

                all_interval_imgs[p] = interval_imgs

        p = p + 1

    print(all_interval_imgs)

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
