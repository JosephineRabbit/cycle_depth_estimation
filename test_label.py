import os

import cv2

path = '/home/dut-ai/Documents/temp/synthia/test/label_test/SEQ6_right_0000194.png'

img = cv2.imread(path)

print(img.max())

