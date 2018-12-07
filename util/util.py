from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F
synthia_txtpath='/home/dut-ai/Documents/temp/synthia_encoding.txt'
cat2color_synthia={}
with open(synthia_txtpath,'r') as file:
    for line in file.readlines():
        templist = line.strip().split('\t')
        label = templist.pop(0)
        templist=[int(element) for element in templist]
        cat2color_synthia[int(label)] = templist

cityscape_txtpath='/home/dut-ai/Documents/temp/cityscape_encoding.txt'
cat2color_cityscape={}
with open(cityscape_txtpath,'r') as file:
    for line in file.readlines():
        templist = line.strip().split('\t')
        label = templist.pop(0)
        templist=[int(element) for element in templist]
        cat2color_cityscape[int(label)] = templist


def label2im(image_tensor):


    cat2color=cat2color_cityscape

    if len(image_tensor.shape)==3:
        print(image_tensor.shape)
        image_tensor=image_tensor.cpu().numpy()[0,:,:]
    else:
        print('++++++++++',image_tensor.shape)
        image_tensor=np.argmax(image_tensor[0,:,:,:].cpu().numpy(),0)
        print('------------',image_tensor.shape)
    h=image_tensor.shape[0]
    w=image_tensor.shape[1]
    image_show=np.zeros(shape=[h,w,3])
    for category in list(cat2color.keys()):
        try:
            x, y = np.where(image_tensor == category)
            image_show[x, y] = np.array(cat2color[category])
        except:
            continue
    return image_show

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(1)
    else:
        return input_image

    image_numpy = image_tensor[0].cpu().float().numpy()

    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    print(image_numpy.shape)
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def scale_pyramid(img, num_scales):
    scaled_imgs = [img]

    s = img.size()

    h = s[2]
    w = s[3]

    for i in range(1, num_scales):
        ratio = 2**i
        nh = h // ratio
        nw = w // ratio
        scaled_img = F.upsample(img, size=(nh, nw), mode='nearest')
        scaled_imgs.append(scaled_img)

    scaled_imgs.reverse()
    return scaled_imgs