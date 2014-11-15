from PIL import Image

file_in = '11_1_.gif'; file_out = 'test.png'
im = Image.open(file_in)
im.save(file_out)

