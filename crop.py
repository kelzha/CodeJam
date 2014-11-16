from PIL import Image
import glob
import os

files = glob.glob("./training_dataset/*.png")


def cropp(imageFile, centerX, centerY, W, H):
    filepath,filename = os.path.split(imageFile)
    filtername,exts = os.path.splitext(filename)
    im = Image.open(imageFile)
            
    #print im.size
    box = (int(centerX - W/2), int(centerY - H/2), int(centerX + W/2), int(centerY + H/2))
    print box
    cropped_im = im.crop(box)
    cropped_im.save('./training_dataset_cropped/'+filtername+'_cropped'+".png",'PNG')