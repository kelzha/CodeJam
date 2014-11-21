import numpy as np
import cv2
import sys
from PIL import Image
import glob
import crop
import aspectRatio
import os

files = glob.glob("./Manual Crop/*")