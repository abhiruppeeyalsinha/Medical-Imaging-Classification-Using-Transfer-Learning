import matplotlib.pyplot as plt
import numpy as np
import cv2,os
import glob

rows = 3
cols = 5
fig = plt.figure(figsize=(7,8))

file_path =r"E:\DICOM\medical dataset\archive\loop test_dyed-resection-margins\*.JPG"
file = (glob.glob(file_path))
img_file = [cv2.imread(img) for img in file]


for i in range(0,len(img_file),rows*cols):
    fig = plt.figure(figsize=(7,8))
    for j in range(0,rows*cols):
        fig.add_subplot(rows,cols,j+1)
        plt.imshow(cv2.cvtColor(img_file[i+j],cv2.COLOR_BGR2RGB))
    plt.show()    

