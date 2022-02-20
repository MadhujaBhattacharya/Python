import cv2 
import numpy as np
from matplotlib import pyplot as plt
image= cv2.imread('house.jpg')
row, column= image.shape[:2]
kernel_x= cv2.getGaussianKernel(column, 250)
kernel_y= cv2.getGaussianKernel(row,250)
kernel= kernel_y*kernel_x.T
filter=255*kernel/np.linalg.norm(kernel)
vintage_image= np.copy(image)

for i in range(3):
    vintage_image[:,:,i]=vintage_image[:,:,i]*filter

plt.imshow(vintage_image)
plt.show()