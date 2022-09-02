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


import registration_utils

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
parser.add_argument(
    "--barcode", type=str, required=True, help="specify barcode to analyze"
)

parser.add_argument(
    "-p",
    "--position_list",
    nargs='*',
    type=str,
    required=False,
    help="specify positions to process. E.g. -p P1 P2"
)



if __name__ == "__main__":
    args = parser.parse_args()

    # Open the file and load the file
    # Open the file and load the file
    barcode = args.barcode

    yaml_dir = os.path.join(args.output_path, "yml_configs")

    yaml_list = [x for x in os.listdir(yaml_dir) if "_confirmed" in x]
    yaml_list = [x for x in yaml_list if barcode in x]
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
    dfconfig.set_index(["barcode", "round"], inplace=True)

    mag = "20x"

    output_dir = dfconfig["output_path"][0]
    csv_dir = output_dir + os.sep + "csvs"
    csv_name = barcode + "cleanedup_match_csv.csv"
    csv_path = csv_dir + os.sep + csv_name
    print("\n\n" + csv_path + "\n\n")
    # dfall = pd.read_csv(csv_path)
    dfall = pd.read_csv(csv_path)


    output_dir = dfconfig["output_path"][0]
    align_csv_dir = output_dir + os.sep + "alignment_csvs_each"
    align_csv_name_glob = f"{barcode}*alignment_csv_each.csv"
    print(align_csv_name_glob)
    globlist = glob(align_csv_dir + os.sep + align_csv_name_glob)
    dfalign_list = []
    for align_csv_path in globlist:
        # align_csv_path = align_csv_dir + os.sep + align_csv_name
        df = pd.read_csv(align_csv_path)
        dfalign_list.append(df)
    dfalign = pd.concat(dfalign_list)

    dfall["parent_file"] = dfall["parent_file"].apply(lambda x: os_swap(x))

    # merge both dataframes so that you only try to align the positions that can be aligned.
    dfall = pd.merge(
        dfalign,
        dfall,
        on=["key", "template_position"],
        suffixes=("_align", ""),
        how="left",
    )

    keylist = dfall["key"].unique()
    # for Position in ['P2']:

    if args.position_list is not None:
        template_position_list = [x for x in np.sort(dfall["template_position"].unique()) if x in args.position_list]
        print('choosing subset of positions = ', args.position_list)
    else:
        template_position_list = np.sort(dfall["template_position"].unique())

    print(template_position_list)
    # go one position by position, since you need offsets per position
    for Position in template_position_list:  

        print("POSITION = ", Position)



        # need to define the keylist for each position, since some positions may not be imaged every round
        keylist = dfall.set_index("template_position").loc[Position, "key"].unique()
        # keylist = [x for x in keylist if "Time" not in x]
        # print(testing_keylist)
        # for ki, key in enumerate(testing_keylist):
        key = '20x_Timelapse' 
        if key not in keylist:
            key = 'Round 1'
            print('no timelapse present: choosing round 1 instead')

    
        print(key)



        # first thing that needs to be found is the reference image (i.e. timelapse image) details
        alignment_offset = eval(
                dfalign.set_index(["key", "template_position"]).loc[
                    pd.IndexSlice[key, Position], "alignment_offsets_zyx"
                ]
            )
        print(alignment_offset)

        

        # then the shape of the aligned images needs to be determined
        initial_shape = [45,1248,1848]
                
        zpad = 20
        xypad = 300
        final_shape = np.uint16(
            np.asarray(
                [
                    initial_shape[0] + zpad*2,
                    initial_shape[1] + xypad*2,
                    initial_shape[2] + xypad*2,
                ]
            )
        )

        
        # initial image size as mimic. The goal is to shift an image of ones in the same way the reference image was shifted so this becomes a mask for cropping
        blank_image = np.ones(initial_shape,dtype='uint16')
        
        
        
        #now compute the alignment transform for the reference image
        align_matrix = registration_utils.get_align_matrix(alignment_offset)
        shift_to_center_matrix = registration_utils.get_shift_to_center_matrix(
            blank_image.shape, final_shape
        )
        combo = shift_to_center_matrix @ align_matrix

        
        # now perform the alignment transform to shift the ones
        transformed_blank = affine_transform(
            blank_image,
            np.linalg.inv(combo),
            output_shape=final_shape,
            order=0,  # order = 0 means no interpolation...juust use nearest neighbor
        )

        # now perform max intensity projection since the upcoming crop will be performed on the max intensity projection exports. 
        transformed_blank_mip = np.max(transformed_blank,axis=0)

        # now identify the crop ROI
        
            
        crops = [0,0,0] #gap between edge of label and edge of roi z,y,x
        py, px = np.nonzero(transformed_blank_mip)
        roi=[]
        for pi,p in enumerate([py,px]):
            roi.extend( [np.max( (p.min()-crops[pi], 0) ), 
                        np.min( (p.max() + crops[pi], transformed_blank_mip.shape[pi]) )] )

        # this index epxression can now be applied to all the images for a given position (rounds and timepoints and channels) to crop
        crop_expression = np.index_exp[roi[0]:roi[1],roi[2]:roi[3]]
        print(crop_expression)

            




        dfr = dfall.set_index(["template_position", "key"])
        dfsub = dfr.loc[pd.IndexSlice[[Position], [key]], :]
        parent_file = dfsub["parent_file"][0]
        scene = dfsub["Scene"][0]

        # # 3 get variables for file name
        # # round_num0 = re.search("time|Round [0-9]+", parent_file, re.IGNORECASE).group(0)
        # search_out = re.search("time|Round [0-9]+", parent_file, re.IGNORECASE)
        # assert search_out is not None  # necessary for passing mypy type errors
        # round_num0 = search_out.group(0)



        scene_num = str(scene).zfill(2)
        position_num = Position[1::].zfill(2)
        well = dfsub["Well_id"][0]

        position = Position
        si = int(scene) - 1  # scene_index


        sep = os.sep
        globdir = (
            f"{output_dir}{sep}mip_exports{sep}{barcode}{sep}"
        )


        # now search for the images to crop
        fovid = f"{barcode}_R*_P{position_num}"
        globname = (
            f"{fovid}-mip-c*_T*.tif"
        )
        globpath = os.path.abspath(f"{globdir}{sep}{globname}")


        # now find all files associated with that position and round number (should be multiple channels and potentially multiple timepoints)
        globlist = glob(globpath)
        # [print(x) for x in globlist]
        print(len(globlist))

        # now iterate through all the files and perform the cropping.
        for imgi,img_path in enumerate(globlist):
            reader = AICSImage(img_path)
            img = np.squeeze(reader.get_image_data())
            crop_img = img[crop_expression]

            # now determine where to save the cropped image
            savedir = (
            f"{output_dir}{sep}mip_exports_tcropped{sep}{barcode}{sep}"
            )
            
            savename = Path(img_path).name
            savepath = savedir + os.sep + savename

        

            if not os.path.exists(savedir):
                os.makedirs(savedir)
                print("making", os.path.abspath(savedir))

            file_name_stem = Path(dfsub["parent_file"].iloc[0]).stem

            assert os.path.abspath(savepath) is not os.path.abspath(img_path)

            skio.imsave(
                savepath, np.uint16(crop_img), check_contrast=False
            )

            if imgi==0:
                print(os.path.abspath(savepath))
