import os
from aicsimageio import AICSImage
import sys
print(sys.version_info)
filename = '//allen/aics/microscopy/Antoine/Analyse EMT/4i Data/5500000724/ZSD1/Round 6/5500000724_Round_6_20X_P1_to_P16.czi'
print(os.path.abspath(filename))

reader = AICSImage(filename)
T=0
ci=0
reader.set_scene(14)
delayed_chunk = reader.get_image_dask_data("ZYX", T=T, C=ci)

print(reader.current_scene)

print(reader.current_scene_index)

imgstack = delayed_chunk.compute()

print(imgstack.shape)