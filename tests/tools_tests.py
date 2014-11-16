from codehams.face_rec.tools import EigenFacial
import numpy as np
from PIL import Image
def test_eigenfacial_OG_sanity():
  import os
  
  data_dir = "C:/Users/Selimb/Documents/yaleB/CroppedYale/CroppedYale"
  os.chdir(data_dir)
  
  def create_filenames(data_dir, view_list):
      # loads the pictures into a list
      # data_dir: the CroppedYale folder
      # view_list: the views you wish to grab
      dir_list = os.listdir(data_dir)
      file_list = []
      for dir in dir_list:
          for view in view_list:
              filename = "%s/%s_%s.png" % (dir, dir, view)
              file_list.append(filename)
      return(file_list)

  view_list = ['P00A+000E+00', 'P00A+005E+10' , 'P00A+005E-10' , 'P00A+010E+00']

  file_list = create_filenames(data_dir, view_list)
  # open image
  im = Image.open(file_list[0]).convert("L")
  # get original dimensions
  H,W = np.shape(im)
  # print 'shape=',(H,W)

  im_number = len(file_list)
  # fill array with rows as image
  # and columns as pixels
  arr = np.zeros([im_number,H*W])

  for i in range(im_number):
      im = Image.open(file_list[i]).convert("L")
      arr[i,:] = np.reshape(np.asarray(im),[1,H*W])

  facial = EigenFacial(arr[:20],file_list[:20])
  scores = facial.train()

  new_score = facial.get_score(arr[0],facial.mean_image)
  assert(np.max(new_score - scores[0]) < 10**-10)
  assert(facial.recognize(arr[0]) == 0)


