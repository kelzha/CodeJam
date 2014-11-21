from PIL import Image
import glob
import os
import numpy as np
import cv2
#files = glob.glob("./training_dataset/*.png")


def cropp(imageFile, rect):
    filepath,filename = os.path.split(imageFile)
    filtername,exts = os.path.splitext(filename)
    im = Image.open(imageFile).convert("L")

    GoldenRto = 1.418

    #print im.size
    #Address blackboxing at bottom
    bottom = rect[3]
    top = rect[1]
    left = rect[0]
    right = rect[2]
    
    if bottom > im.size[1]:
    	delta = bottom - im.size[1]
        top -= delta
    	bottom = im.size[1]

    box = (rect[0], top, rect[2], bottom)
    # print box
    cropped_im = im.crop(box)

    fixedW = 160

    fixedH = int(fixedW * GoldenRto)
    print filtername
    # plt.imshow(cropped_im)
    (cropped_im.resize((fixedW,fixedH),Image.ANTIALIAS)).save("./Crop_edited/" +filtername+"_edited.bmp",'BMP')
    '''image = cv2.imread("./participantdataset_edited/" +filtername+"_edited.bmp", 0)
    equ = cv2.equalizeHist(image)
    im = Image.fromarray(equ)
    im.save("./participantdataset_edited/" +filtername+"_edited.bmp",'BMP')'''
