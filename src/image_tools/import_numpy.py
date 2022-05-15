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

#from src.image_tools.filter_ids import filter_ids
#from src.image_tools.remove_ids import remove_ids

from filter_ids import filter_ids
from remove_ids import remove_ids

def import_numpy_2d(path, clinical_ids, random_crop=True, crop_size=(512, 512)):
    img_array = np.load(path)

    img_array = filter_ids(img_array, clinical_ids)

    if random_crop:
        cropped_array = np.empty(shape=(img_array.shape[0], crop_size[0]*crop_size[1]+1), dtype=np.int8)

        i = 0 
        for image in img_array:
            id = image[-1]

            image = remove_ids(image)
            
            image = np.reshape(image, (-1, int(sqrt(len(image)))))

            image = tf.image.random_crop(value=image, size=crop_size)

            image = image.numpy()

            image = image.flatten()

            image = np.append(image, id)

            cropped_array[i] = image

            i = i + 1

        img_array = cropped_array

    return img_array

def import_numpy_3d(dir):
    patients = os.listdir(dir)
    for patient in patients:
        patient = os.path.join(dir, patient)
        patient_mesh = mesh.Mesh.from_file(patient)
        vertices = np.around(np.unique(patient_mesh.vectors.reshape(int(patient_mesh.vectors.size/3), 3), axis=0),2)
        patient_mesh = np.asarray(patient_mesh)
        # for instance, if patient mesh shape is (375030, 9),
        # there are 375030 faces and each face has three vertices with 
        # three coordinate values each. Therefore, vertice count is equal to:
        vertice_count = patient_mesh.shape[0]*np.sqrt(patient_mesh.shape[1])

        tri_obj = Trimesh()
        patient_mesh_tri = trimesh.load_mesh(patient, enable_post_processing=True, solid=True)
        faces=patient_mesh_tri.faces
        print(vertices)
        print(faces)
        tri_obj.vertices=vertices
        tri_obj.faces=faces
        #grid = creation.voxelize(mesh=tri_obj, pitch=1)
        #print(grid)

import_numpy_3d("C:\\Users\\trist\\cs_projects\\Cancer_Project\\Cancer Imagery\\3d_duke")
