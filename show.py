from PIL import Image
import matplotlib.pyplot as plt

#Convert GIF to PNG
file_in = '11_1_.gif'; file_out = 'test.png'
gif = Image.open(file_in)
gif.save(file_out)

#Load image and show
im = cv2.imread(file_out)
plt.imshow(im)