from PIL import Image
from os import listdir
from os.path import isfile, join

src_path = "./src/"
dst_path = "./dst/"
im_files = [f for f in listdir(src_path) if isfile(join(src_path, f))]

new_w = 1600
new_h = 1600
mx, my = new_w//2, new_h//2

for im_file in im_files:
    try:
      print(im_file)
      src_image = Image.open(src_path + im_file)
      w, h = src_image.size
      new_image = Image.new('RGB', (new_w, new_h))
      x = (new_w - w) // 2
      y = (new_h - h) // 2
      new_image.paste(src_image, (x, y))
      new_image.save(dst_path + im_file)
    except Exception as e: 
      print(e)
      pass
