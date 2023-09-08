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
parser.add_argument("--input_path", type=str, default="/allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports/mip_exports_v2/5500000728-export", help="input path that has the positions that performed badly for the timelapse")

parser.add_argument("--output_path", type=str, default="/allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports/mip_exports_v2/5500000728", help="output path that contains the rest of the positions",)


if __name__ == "__main__":
    args= parser.parse_args()
