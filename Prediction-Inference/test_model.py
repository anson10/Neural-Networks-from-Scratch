import cv2

image_data = cv2.imread('tshirt.png' , cv2.IMREAD_GRAYSCALE)

import matplotlib.pyplot as plt
plt.imshow(image_data, cmap = 'gray' )
plt.show()

image_data = cv2.resize(image_data, (28, 28 ))

plt.imshow(image_data, cmap = 'gray' )
plt.show()