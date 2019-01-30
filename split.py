import random
import os
import shutil
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
path ='/home/dut-ai/Documents/depthdataset/indoor/all/opengl/test/dep'
files = os.listdir(path)

for file in files:
    dep  = np.load(path+'/'+file)
    #print(dep.max(),dep.min())


    #dep.show()
    dep_np = np.array(dep)
    print(dep_np.max(),dep_np.min())







a = 0
if a==1:
    re_ldsit = []
    A = 0
    B = 8403
    COUNT = 200
    path_dep = '/home/dut-ai/Documents/vt_kitti/depth_all'
    out_dep = '/home/dut-ai/Documents/vt_kitti/depth_all_test/'
    p_i = '/home/dut-ai/Documents/vt_kitti/vkitti_1.3.1_rgb/all'
    out_i = '/home/dut-ai/Documents/vt_kitti/vkitti_1.3.1_rgb/all_test/'
    p_seg = '/home/dut-ai/Documents/vt_kitti/vkitti_1.3.1_scenegt/all'
    out_seg = '/home/dut-ai/Documents/vt_kitti/vkitti_1.3.1_scenegt/all_test/'
    files = os.listdir(path_dep)

    re_ldsit = random.sample(range(A, B + 1), COUNT)

    for i in re_ldsit:
        shutil.move(path_dep + '/' + files[i], out_dep + files[i])
        shutil.move(p_i + '/' + files[i], out_i + files[i])
        shutil.move(p_seg + '/' + files[i], out_seg + files[i])






