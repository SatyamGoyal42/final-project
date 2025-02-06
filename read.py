import cv2 as cv
img  = cv.imread('images/Baboon_Gray.png')



def rescaleFrame(frame,scale):
    w  = int(frame.shape[0]*scale)
    h  = int(frame.shape[1]*scale)

    dimensions  = (w,h)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

resized_img = rescaleFrame(img,.5)
cv.imshow("baboonsmall",resized_img)
cv.imshow("baboon",img)
cv.waitKey(0)
