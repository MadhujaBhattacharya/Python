import cv2
import matplotlib.pyplot as plt
image= cv2.imread('image.jpg')
grey_image= cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT)
plt.imshow(grey_image)
plt.show()