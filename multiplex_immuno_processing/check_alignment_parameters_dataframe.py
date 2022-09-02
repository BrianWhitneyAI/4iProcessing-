import argparse
from glob import glob
import os
from pathlib import Path
import re

from aicsimageio import AICSImage
import numpy as np
import pandas as pd
from scipy.ndimage import affine_transform
import skimage.io as skio
import yaml
from yaml.loader import SafeLoader

overwrite = True



def os_swap(x):
    out = "/" + ("/".join(x.split("\\"))).replace("//", "/")
    return out




parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="output dir of all processing steps. This specifies where to find the yml_configs too",
)


parser.add_argument("--method",
            choices=['cross_cor', 'ORB']
)


class Args:
    output_path = "//allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports"
    # barcode = '5500000724'
    
    

if __name__ == "__main__":
    args = parser.parse_args()


    # Open the file and load the file
    # Open the file and load the file
    # barcode = args.barcode

    yaml_dir = os.path.join(args.output_path, "yml_configs")

    yaml_list = [x for x in os.listdir(yaml_dir) if "_confirmed" in x]
    # yaml_list = [x for x in yaml_list if barcode in x]
    dflist = []
    for y in yaml_list:
        print(y)
        yml_path = yaml_dir + os.sep + y
        with open(yml_path) as f:
            data = yaml.load(f, Loader=SafeLoader)
            for round_dict in data["Data"]:
                dfsub = pd.DataFrame(round_dict.values(), index=round_dict.keys()).T
                dfsub["barcode"] = data["barcode"]
                dfsub["scope"] = data["scope"]
                dfsub["output_path"] = data["output_path"]
                dflist.append(dfsub)

    dfconfig = pd.concat(dflist)
    dfconfig.set_index(["barcode"], inplace=True)

    for barcode in np.unique(dfconfig.index.values):
        mag = "20x"

        
        # output_dir = dfconfig["output_path"][0]
        output_dir = dfconfig.loc[barcode,'output_path'][0]
        csv_dir = output_dir + os.sep + "csvs"
        csv_name = barcode + "cleanedup_match_csv.csv"
        csv_path = csv_dir + os.sep + csv_name
        print("\n\n" + csv_path + "\n\n")
        # dfall = pd.read_csv(csv_path)
        dfall = pd.read_csv(csv_path)


        output_dir = dfconfig["output_path"][0]
        align_csv_dir = output_dir + os.sep + "alignment_csvs_each_" + args.method
        align_csv_name_glob = f"{barcode}*alignment_csv_each.csv"
        print(align_csv_name_glob)
        globlist = glob(align_csv_dir + os.sep + align_csv_name_glob)
        dfalign_list = []
        for align_csv_path in globlist:
            # align_csv_path = align_csv_dir + os.sep + align_csv_name
            df = pd.read_csv(align_csv_path)

            keepcols = [x for x in df.columns.tolist() if not bool(re.search('unnamed',x,re.IGNORECASE))]
            dfalign_list.append(df[keepcols])
        if len(dfalign_list)>0:
            dfalign = pd.concat(dfalign_list)

            dfall["parent_file"] = dfall["parent_file"].apply(lambda x: os_swap(x))

            # merge both dataframes so that you only try to align the positions that can be aligned.
            # first identify missing positions
            position_all_list = [x for x in dfall['template_position'].tolist()]
            position_align_list = [x for x in dfalign['template_position'].tolist()]
            missing_position_list = np.unique([x for x in position_all_list if x not in position_align_list] )
            print(f"barcode = {barcode}")
            print(f"number of positions without computed alignment parameters = {len(missing_position_list)}")
            print(f"missing_position_list = {missing_position_list}")
            print()

            # now find positions that failed alignment
            print('positions that failed alignment')
            print(dfalign[dfalign['failed']])
            
            # now look for positions with very large shifts
            # first determine x y z values
            xypad = 200
            for i, zyx in enumerate('zyx'):
                dfalign[zyx] = [np.abs(eval(x)[i]) for x in dfalign['alignment_offsets_zyx'].tolist()]
            print()
            print('positions that have very large shifts')
            print(dfalign[(dfalign['x']>xypad)|(dfalign['y']>xypad)|(dfalign['z']>20)])
                
        else:
            print(f'NO ALIGNMENT PERFORMED FOR barcode = {barcode}')