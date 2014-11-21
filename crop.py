from PIL import Image
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

files = glob.glob("./training_dataset/*.png")


def cropp(imageFile, centerX, centerY, W, H):
    filepath,filename = os.path.split(imageFile)
    filtername,exts = os.path.splitext(filename)
    im = Image.open(imageFile).convert("L")
    #print im.size
    #Address blackboxing at bottom
    bottom = int(centerY + H/2.0)
    top = int(centerY - H/2.0)

    if bottom > im.size[1]:
    	delta = bottom - im.size[1]
        top -= delta
    	bottom = im.size[1]


    box = (int(centerX - W/2.0), top, int(centerX + W/2.0), bottom)
    # print box
    cropped_im = im.crop(box)
    # plt.imshow(cropped_im)
    return cropped_im
    #cropped_im.save("edited_" +filtername+".png",'PNG')
