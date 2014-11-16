import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob

files = glob.glob("./training_dataset/*7_.png")

for file in files:
    img2 = cv2.imread(file,0)
    img = cv2.GaussianBlur(img2,(1,15),5)
    # Initiate STAR detector
    orb = cv2.ORB()

    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # print kp

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    xx = [i.pt for i in kp]
    pointset = np.array(xx)
    # print pointset
    # print pointset[:,0].shape
    # print pointset
    yup = np.min(pointset[:,1])
    xleft = np.min(pointset[:,0])
    ydown = np.max(pointset[:,1])
    xright =np.max(pointset[:,0])

    # draw only keypoints location,not size and orientation
    img3 = cv2.drawKeypoints(img2,kp,color=(0,255,0), flags=0)
    # cv2.rectangle(img, (10, 10), (20, 20), (0, 255, 0))
    cv2.rectangle(img3, (int(xleft), int(yup)), (int(xright), int(ydown)), (0, 255, 0))
    plt.imshow(img3),plt.show()
    # cv2.rectangle(img, (10, 10), (20, 20, (0, 255, 0))

    # cv2.imshow('img',img)
