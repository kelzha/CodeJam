import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

im = Image.open('training_dataset/5_9_.gif')
im.save('test.png')
a = cv2.imread('test.png')

np.set_printoptions(threshold='nan')
print(a)

