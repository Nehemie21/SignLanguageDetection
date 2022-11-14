import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector

capture = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)

offset = 20
imgSize = 421

folder = "Data/A"
counter = 0
while True:

    success, img = capture.read()
    hands, img = detector.findHands(img)


    if hands:
        hand = hands[0]
        x,y,width, height = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)
        imgCrop = img[y - offset:y+height+offset,
                 x-offset:x+width+offset]
      #  imgCrop = img[y:y+height,
               #     x:x+width]

        imgCropShape = imgCrop.shape

#        imgWhite[0:imgCropShape[0], 0:imgCropShape[1]] = imgCrop

        aspectRatio = height/width

        if aspectRatio > 1:
            constant = imgSize/height
            widthCalculated = math.ceil(constant*width)
            imgResize = cv2.resize(imgCrop, (widthCalculated, imgSize))
            imgResizeShape = imgResize.shape
            widthGap = math.ceil((imgSize - widthCalculated)/2)
            imgWhite[:, widthGap:widthCalculated + widthGap] = imgResize
            # print("img Resize shape", imgResize.shape)
            # print("img White shape", imgWhite.shape)
        else:
            constant = imgSize/width
            heightCalculated = math.ceil(constant*height)
            imgResize = cv2.resize(imgCrop, (imgSize, heightCalculated))
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((imgSize - heightCalculated)/2)
            imgWhite[heightGap:heightCalculated + heightGap, :] = imgResize
            # print("img Resize shape", imgResize.shape)
            # print("img White shape", imgWhite.shape)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        #data = cv2.imread("asl")




    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpeg', imgWhite)
        print(counter)

#def findSign(self):


