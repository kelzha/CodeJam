import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import crop
import numpy as np
import cv2
import faceDetection

def locate_face(file_path):
    filepath,filename = os.path.split(file_path)
    filtername,exts = os.path.splitext(filename)
#    testIm = Image.open(file_path).convert('L')
    H=100
    W=100
    AR=1
    
    testIm=crop.cropp(file_path,faceDetection.getRectangle(file_path))

#    mean_files = glob.glob('C:/Users/Doug/Documents/GitHub/codehams/participantdataset_edited/*.*')
#    mean_face_imgfmt = [None] * len(mean_files)
#    mean_face_W= [None] * len(mean_files)
#    mean_image= [None] * len(mean_files)
#    for i in xrange(0,len(mean_files)):
#        mean_face_imgfmt[i]=Image.open(mean_files[i]).convert('L')
#        mean_image[i]=np.reshape(np.asarray(mean_face_imgfmt[i]),[1,H*W])
#
#    mean_face_W,mean_face_H=mean_face_imgfmt[0].size
    mean_face_W=100
    mean_face_H=100
    
    def cropSection(actualImage,posX,posY,cropX,cropY,AR):
        bigX,bigY=cropX,cropY
        if (1.0*bigX/bigY)>AR:
            newY=bigY
            while (1.0*newY*AR)%1 !=0.0: 
                newY=newY-1 [Errno 13] Permission denied:
            newX=int(newY*AR)
        else:
            newX=bigX
            while (1.0*newX/AR)%1 !=0.0: 
                newX=newX-1
            newY=int(1.0*newX/AR)
        return actualImage.crop((posX,posY,posX+newX,posY+newY))
        
    i=0
    j=0
    testIm_W,testIm_H=testIm.size
    decay=0.6
    min_norm=9999999999999
    
    kmax=8
    for k in xrange(1,kmax):
        testIm_c=cropSection(testIm,0,0,int(testIm_W/k**decay),int(testIm_H/k**decay),AR)
        testIm_c_W,testIm_c_H=testIm_c.size
        dxRight=testIm_W-testIm_c_W
        dxBot=testIm_H-testIm_c_H
        print 'testIm_c_W,testIm_c_H', testIm_c_W,testIm_c_H,' dxRight,dxBot,k',dxRight,dxBot,k
        
    for k in xrange(1,kmax):
        print k,
        testIm_c=cropSection(testIm,0,0,int(testIm_W/k**decay),int(testIm_H/k**decay),AR)
        testIm_c_W,testIm_c_H=testIm_c.size
        #print testIm_c.size
        #plt.imshow(testIm_c)
        #plt.figure()
        dxRight=testIm_W-testIm_c_W
        dxBot=testIm_H-testIm_c_H
        step=20
        xSteps=step
        ySteps=step
        if(dxRight==0): 
            dxRight=dxRight+1
            xSteps=1
        if(dxBot==0): 
            dxBot=dxBot+1
            ySteps=1
        if(dxRight/xSteps==0):xSteps=1
        if(dxBot/ySteps==0):ySteps=1

        #print dxRight,dxRight/xSteps
        #print dxBot,dxBot/ySteps
        for i in xrange(0,dxRight,dxRight/xSteps):
            for j in xrange(0,dxBot,dxBot/ySteps):
                testIm_c=cropSection(testIm,i,j,int(testIm_W/k**decay),int(testIm_H/k**decay),AR)
                testIm_c_W,testIm_c_H=testIm_c.size
                
                #print testIm_c.size
                #plt.imshow(testIm_c)
                #plt.figure()
    
                testIm_cs=testIm_c.resize((mean_face_W,mean_face_H), Image.ANTIALIAS)
                testIm_cs.save("C:/Users/Doug/Desktop/McGill Fall 2014/CODEJAM/doug_test1/doug_"+filtername +".png",'PNG')
#                testIm_cs_arr=(np.reshape(np.array(testIm_cs),[1,H*W]))
#                standard_dev=np.std(testIm_cs_arr)
#
#                current_norm=0
#                if(standard_dev>30.0):
#                    for m in xrange(0,len(mean_files)):
#                        min_mean_norm=9999999999999
#                        min_m=0
#                        testIm_cs_norm=testIm_cs_arr-mean_image[m]
#                        current_norm=np.linalg.norm(testIm_cs_norm)
#                        if(current_norm<min_mean_norm):
#                            min_mean_norm=current_norm
#                            min_m=m                   
#                    if(min_mean_norm<min_norm):
#                        min_norm=min_mean_norm
#                        min_i=i
#                        min_j=j
#                        min_k=k


    xSize=int(testIm_W/min_k**decay)
    ySize=int(testIm_H/min_k**decay)
    testIm_c=cropSection(testIm,min_i,min_j,xSize,ySize,AR)
    testIm_c_W,testIm_c_H=testIm_c.size
     
    testIm_cs=testIm_c.resize(mean_face_imgfmt[0].size, Image.ANTIALIAS)
#    testIm_cs.save("C:/Users/Doug/Desktop/McGill Fall 2014/CODEJAM/doug_test1/doug_"+filtername +".png",'PNG')
    testIm_cs_arr=np.asarray(testIm_cs)
#    testIm_immm=Image.fromarray(cv2.equalizeHist(testIm_cs_arr))
#    testIm_immm.save("dougHist_"+filtername +".png",'PNG')
    testIm_cs_norm=(testIm_cs_arr-np.mean(mean_image))
    print min_norm,min_i,min_j,min_k
    print np.linalg.norm(testIm_cs_norm)

    centerX=min_i+xSize/2
    centerY=min_j+ySize/2
    Wi=xSize
    He=ySize
    print " "
    return centerX, centerY, Wi, He
    
dataset=glob.glob('C:/Users/Doug/Desktop/McGill Fall 2014/CODEJAM/Participant_Dataset-2014-11-20/Codehams/260396387_2_.bmp')
mean_face_imgfmt = [None] * len(dataset)
kkk=0
for pic_path in dataset:
    kkk=kkk+1
    print 'pic#', kkk
#pic_path='C:/Users/Doug/Desktop/McGill Fall 2014/CODEJAM/barack_obama_56.jpg'
    centerX, centerY, Wi, He =locate_face(pic_path)
#    crop.cropp(pic_path, centerX, centerY, Wi, He)