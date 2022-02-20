import cv2
import matplotlib.pyplot as plt
image= cv2.imread('house.jpg')
edges= cv2.Canny(image,100,300)
plt.imshow(edges)
plt.show()