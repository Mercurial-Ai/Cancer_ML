import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def grey_to_rgb(img_array):
    width = img_array.shape[-3]
    height = img_array.shape[-2]
    new_img_array = torch.empty((img_array.shape[0], img_array.shape[1], width, height, 3), dtype=torch.uint8).to(device)
    i = 0
    for p in img_array:
        new_p = torch.empty((p.shape[0], width, height, 3), dtype=torch.float16).to(device)
        j = 0
        for s in p:
            s = torch.squeeze(s)
            out = torch.empty((width, height, 3), dtype=torch.float16).to(device)
            out[:, :, 0] = s
            out[:, :, 1] = s
            out[:, :, 2] = s

            new_p[j] = out

            j = j + 1

        new_img_array[i] = new_p

        i = i + 1
    
    return new_img_array
        