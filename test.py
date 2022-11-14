import cv2
import numpy as np
import math
import time
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

capture = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)

classifier = Classifier("C:\\Users\\Nehemie\\PycharmProjects\\SignLanguageDetector\\Model\\keras_model.h5",
                        "C:\\Users\\Nehemie\\PycharmProjects\\SignLanguageDetector\\Model\\labels.txt")
offset = 21
imgSize = 421

folder = "Data/A"
counter = 0

# labels = ["asl: a", "asl: b", "asl: c", "asl: d", "asl: e",
#           "asl: f", "asl: g", "asl: h", "asl: i", "asl: j",
#           "asl: k", "asl: l", "asl: m", "asl: n", "asl: o",
#           "asl: p", "asl: q", "asl: r", "asl: s", "asl: t",
#           "asl: u", "asl: v", "asl: w", "asl: x", "asl: y",
#           "asl: z", "asl: 0", "asl: 1", "asl: 2","asl: 3",
#           "asl: 4", "asl: 5", "asl: 6", "asl: 7", "asl: 8",
#           "asl: 9"]

labels = ["a", "b", "c", "d", "e",
          "f", "g", "h", "i", "j",
          "k", "l", "m", "n", "o",
          "p", "q", "r", "s", "t",
          "u", "v", "w", "x", "y",
          "z", "0", "1", "2","3",
          "4", "5", "6", "7", "8",
          "9"]
while True:

    success, img = capture.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)


    if hands:
        hand = hands[0]
        x,y,width, height = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
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
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)
            # print("img Resize shape", imgResize.shape)
            # print("img White shape", imgWhite.shape)
        else:
            constant = imgSize/width
            heightCalculated = math.ceil(constant*height)
            imgResize = cv2.resize(imgCrop, (imgSize, heightCalculated))
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((imgSize - heightCalculated)/2)
            imgWhite[heightGap:heightCalculated + heightGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw = False)
            # print("img Resize shape", imgResize.shape)
            # print("img White shape", imgWhite.shape)
            print(prediction, index)


        #cv2.imshow("ImageCrop", imgCrop)
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + width+offset, y + height+offset), (255, 0, 255), 4)


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
        #data = cv2.imread("asl")




    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    # if key == ord("s"):
    #     counter += 1
    #     cv2.imwrite(f'{folder}/Image_{time.time()}.jpeg', imgWhite)
    #     print(counter)

#def findSign(self):


