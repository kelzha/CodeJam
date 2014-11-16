from PIL import Image
import glob
import os

files = glob.glob("./training_dataset/*.png")


def cropp(imageFile, centerX, centerY, W, H):
    filepath,filename = os.path.split(imageFile)
    filtername,exts = os.path.splitext(filename)
    im = Image.open(imageFile)
    #print im.size

    #Address blackboxing at bottom
    bottom = int(centerY + H/2)
    top = int(centerY - H/2)

    if bottom > im.size[1]:
    	delta = bottom - im.size[1]
        top -= delta
    	bottom = im.size[1]

    box = (int(centerX - W/2), top, int(centerX + W/2), bottom)
    print box
    cropped_im = im.crop(box)
    cropped_im.save(filepath + "_cropped/" +filtername+'_cropped'+".png",'PNG')