import cv2

img = cv2.imread('C:/workspace/lotte/test_new/test_new/00302')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


cv2.imshow('H channel', hsv_img[:,:,0])
cv2.imshow('S channel', hsv_img[:,:,1])
cv2.imshow('V channel', hsv_img[:,:,2])
cv2.waitKey()
cv2.destroyAllWindows()