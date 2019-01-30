import os
import cv2
#import h5py
import numpy as np
import os
import h5py
import numpy as np
from PIL import Image

import scipy.io as scio



A = False
if A:


    mat_file_path = '/home/dut-ai/Documents/depthdataset/NYU2/data/nyu_depth_v2_labeled.mat'

    f =h5py.File(mat_file_path, 'r')
    print(f.keys())
    labels = f['labels'].value
    print('--------------',len(labels))
    #f.close()
    labels = np.transpose(labels, [0,2,1])
    labels = labels[:,44:471, 40:601]
    num = labels.shape[0]
    datas = np.load('/home/dut-ai/Documents/depthdataset/NYU2/data/colormap.npy')


    for data in datas:
        print(len(data))
        print('--',data)
    for i in range(num):
        print(labels[i].max(),labels[i].min())




B = True
if B:




    root_path = '/home/dut-ai/Documents/depthdataset/opengl'
    #mat_file_path = os.path.join(root_path, 'data/nyu_depth_v2_labeled.mat')

    image_path = os.path.join(root_path, 'data/image/')
    label_path = os.path.join(root_path, 'data/label/')
    label_raw_path = os.path.join(root_path, 'data/label_raw/')
    depth_path = os.path.join(root_path, 'data/depth/')
    colormap_path = os.path.join(root_path, 'data/colormap.npy')

    if not os.path.isdir(label_path):
        os.mkdir(label_path)
    if not os.path.isdir(label_raw_path):
        os.mkdir(label_raw_path)
    if not os.path.isdir(depth_path):
        os.mkdir(depth_path)
    if not os.path.isdir(image_path):
        os.mkdir(image_path)
    # construct color map
    colormap = np.zeros([256,1,3])
    color_range = np.arange(1,256)
    np.random.shuffle(color_range)
    colormap[1:,0,0] = color_range
    np.random.shuffle(color_range)
    colormap[1:,0,1] = color_range
    np.random.shuffle(color_range)
    colormap[1:,0,2] = color_range
    np.save(colormap_path, colormap)


    img_path = '/home/dut-ai/Documents/depthdataset/opengl/opengl'
    seg_path = '/home/dut-ai/Documents/depthdataset/opengl/semantic'

    i = 0
    for root, dirs, files in os.walk("/home/dut-ai/Documents/depthdataset/opengl/opengl_dep", topdown=False):
        print(root[:43])
        seg_root = seg_path+root[53:]
        img_root = img_path+root[53:]
        A = True
        if A:

            for name in files:
                print(name)
                print(os.path.join(root, name))
                if name[-3:]=='png':
                    i +=1
                    print(i,'----------------------------------------------------')
                    seg_name  = name[:7]+'category40.png'
                    img_name = name[:7]+'color.jpg'

                    print('seg',os.path.join(seg_root,seg_name))

                    seg = cv2.imread(os.path.join(seg_root, seg_name),0)
                    dep = Image.open(os.path.join(root,name))
                    dep = np.array(dep)
                    img = cv2.imread(os.path.join(img_root,img_name))
                    #print('img',os.path.join(img_root,img_name))
                    #print(img.shape)
                    image = img[44:471, 40:601,:]
                    depth = dep[44:471, 40:601]
                    label = seg[44:471, 40:601]


                    print(seg.max())
                    max_class = 39

                    if i%200 == 0:
                        print('%d'%i)
                    np.save(label_raw_path + '%05d' % (i + 1), label)
                    label = np.repeat(label[...,np.newaxis], 3, 2)
                    print(label.shape)
                    label = cv2.LUT(np.uint8(label/max_class*255), colormap)
                    #cv2.cvtColor(label, cv2.COLOR_RGB2BGR)
                    #label = np.transpose(label, [2,1,0])
                    cv2.imwrite(image_path+'%05d.png'%(i+1), image[:,:,-1::-1])
                    cv2.imwrite(label_path+'%05d.png'%(i+1), label)

                    np.save(depth_path + '%05d'%(i+1), depth/1000.)
                 #   np.save(label_raw_path + '%04d'%(i+1), label)


