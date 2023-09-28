import argparse
import os
from aicsimageio import AICSImage
import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader
import registration_utils
import argparse
import shutil


parser = argparse.ArgumentParser()

# Moves the round zero mip exports from ORB(which for some positions performed better) to the output path
# This should only do this for round 0 of the bad positions
# 
parser.add_argument("--input_path", type=str, default="/allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports/mip_exportstest_ORB/5500000728-export", help="input path that has the positions that performed badly for the timelapse")

parser.add_argument("--output_path", type=str, default="/allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports/mip_exports_v2/5500000728", help="output path that contains the rest of the positions",)

#parser.add_argument("--position", type=str, required=True, help="position")

if __name__ == "__main__":
    args= parser.parse_args()
    filenames=[f for f in os.listdir(args.input_path) if "R00" in f]
    print(filenames)
    for i in range(len(filenames)):
        shutil.copy2(os.path.join(args.input_path, filenames[i]), args.output_path)
        



