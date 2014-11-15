from PIL import Image
import glob
import os

files = glob.glob("./training_dataset/*.png")


def cropp(AR,centerX,centerY,W,H):
    for imageFile in files:
        filepath,filename = os.path.split(imageFile)
        filtername,exts = os.path.splitext(filename)
        im = Image.open(imageFile)
        
        if H>W:
            W=AR*H
        else:
            H=W/AR
            
        #print im.size
        box = ((centerX-W/2),(centerX-W/2),(centerY+H/2),(centerY+H/2))
        #print box
        cropped_im=im.crop(box)
        cropped_im.save('./training_dataset_cropped/'+filtername+'_cropped'+".png",'PNG')



cropp(1,150,100,60,200)