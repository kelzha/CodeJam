from PIL import Image
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

#files = glob.glob("./training_dataset/*.png")


def cropp(imageFile, rect):
    filepath,filename = os.path.split(imageFile)
    filtername,exts = os.path.splitext(filename)
    im = Image.open(imageFile).convert("L")

    GoldenRto = 1.618

    #print im.size
    #Address blackboxing at bottom
    bottom = rect[4]
    top = rect[2]

    if bottom > im.size[1]:
    	delta = bottom - im.size[1]
        top -= delta
    	bottom = im.size[1]


    box = (rect[0], top, rect[3], bottom)
    # print box
    cropped_im = im.crop(box)

    fixedW = 160

    fixedH = fixedW * GoldenRto



    # plt.imshow(cropped_im)
    return im.resize((fixedW,fixedH),Image.ANTIALIAS)
    #cropped_im.save("edited_" +filtername+".png",'PNG')
