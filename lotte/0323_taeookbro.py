import numpy as np
import cv2
import pandas as pd


dir = 'C:/workspace/lotte/train/'

categories = []

for i in range(0,1000):
    i = '%d'%i
    categories.append(i)

nb_classes = len(categories) #1000개


polder=[]

for idx, ct in enumerate(categories):

    image_dir = dir+'/'+ct # C:/Users/ai/Desktop/lotte/train/0
    polder = image_dir+'/36.jpg' # C:/Users/ai/Desktop/lotte/train/0/36.jpg
    img = cv2.imread(polder)
    rows, cols = img.shape[:2]

    pts1 = np.float32([[0,-1], [0,rows], [cols, -1],[cols,rows]])
    pts2 = np.float32([[0,70], [100, rows], [cols, 70],[cols-50,rows]])
                  #왼쪽상단 #왼쪽하단    #오른쪽상단 #오른쪽하단


    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective = cv2.warpPerspective(img, matrix, (cols, rows))
    perspective = perspective[70:230,50:240].copy()
    #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ이미지 저장ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
    cv2.imwrite(image_dir+'/48.jpg', perspective)
    #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ이미지 보기ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
    cv2.imshow(ct,perspective)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ이미지 저장ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
    cv2.imwrite()