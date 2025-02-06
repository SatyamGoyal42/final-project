import cv2 as cv
import numpy as np

img = cv.imread("images/Baboon_Gray.png")
blank = np.zeros((500,500,3),dtype='uint8')
blank[:] = 0,255,0

cv.imshow("blank",blank)
cv.imshow("baboon",img)

cv.waitKey(0)