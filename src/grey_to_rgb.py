import numpy as np

def grey_to_rgb(img_array):
    width = img_array.shape[-3]
    height = img_array.shape[-2]
    new_img_array = np.empty((img_array.shape[0], img_array.shape[1], width, height, 3), dtype=np.uint8)
    i = 0
    for p in img_array:
        new_p = np.empty((p.shape[0], width, height, 3), dtype=np.float16)
        j = 0
        for s in p:
            s = np.squeeze(s)
            out = np.empty((width, height, 3), dtype=np.float16)
            out[:, :, 0] = s
            out[:, :, 1] = s
            out[:, :, 2] = s

            new_p[j] = out

            j = j + 1

        new_img_array[i] = new_p

        i = i + 1
    
    return new_img_array
        