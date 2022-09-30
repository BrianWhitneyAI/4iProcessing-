import argparse
from glob import glob
import os
from pathlib import Path
import re
import registration_utils
from aicsimageio import AICSImage
import numpy as np
import pandas as pd
from scipy.ndimage import affine_transform
import skimage.io as skio
import yaml
from yaml.loader import SafeLoader
import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    "--barcode", type=str, required=True, help="specify barcode to analyze"
)

parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="output dir of all processing steps. This specifies where to find the yml_configs too",
)

parser.add_argument(
    "--position",
    type=str,
    required=True,
    help="position"
)

parser.add_argument("--method", choices=['cross_cor', 'ORB', 'merged'])

parser.add_argument(
    "--test_save",
    type=str,
    required=False,
    help="modify output directory by appending 'test' (default='')",
    default = '',
)

if __name__ == "__main__":
    args = parser.parse_args()


    

    # dfall = pd.read_csv(args.dfall_csv_dir)
    mag = "20x"

    Position = args.position

    yaml_dir = os.path.join(args.output_path, "yml_configs")
    barcode = args.barcode

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


    output_dir = dfconfig["output_path"][0]
    align_csv_dir = output_dir + os.sep + "alignment_csvs_each_" + args.method
    align_csv_name = f"{args.barcode}-{args.position}-alignment_csv_each.csv"
    align_csv_path = align_csv_dir + os.sep + align_csv_name


    dfalign = pd.read_csv(align_csv_path)



    output_dir = dfconfig["output_path"][0]
    csv_dir = output_dir + os.sep + "csvs"
    csv_name = barcode + "cleanedup_match_csv.csv"
    csv_path = csv_dir + os.sep + csv_name
    print("\n\n" + csv_path + "\n\n")
    dfall = pd.read_csv(csv_path)


    # dfall["parent_file"] = dfall["parent_file"].apply(lambda x: os_swap(x))

    # merge both dataframes so that you only try to align the positions that can be aligned.
    dfall = pd.merge(
        dfalign,
        dfall,
        on=["key", "template_position"],
        suffixes=("_align", ""),
        how="left",
    )
    



    keylist = dfall.set_index("template_position").loc[args.position, "key"].unique()
    
    count=0
    for ki, key in enumerate(keylist):

        print(f"On Image {count} of {keylist}")
        print(key)
        count+=1

        dfr = dfall.set_index(["template_position", "key"])
        dfsub = dfr.loc[pd.IndexSlice[[Position], [key]], :]

        if args.method == "merged":
            alignment_offset = eval(
                dfall.set_index(["key", "template_position"]).loc[
                    pd.IndexSlice[key, Position], "best_alignment_params"
                ]
            )

        else:
            alignment_offset = eval(
                dfall.set_index(["key", "template_position"]).loc[
                    pd.IndexSlice[key, Position], "alignment_offsets_zyx"
                ]
            )
        print(alignment_offset)
        print(type(alignment_offset))
        final_shape = np.uint16(
            np.asarray(
                [
                    100,
                    1248 + 1248 / 3,
                    1848 + 1848 / 3,
                ]
            )
        )

        parent_file = dfsub["parent_file"][0]
        reader = AICSImage(parent_file)

        scene = dfsub["Scene"][0]

        # 3 get variables for file name
        # round_num0 = re.search("time|Round [0-9]+", parent_file, re.IGNORECASE).group(0)
        
        round_num = str(dfr.loc[pd.IndexSlice[Position,key],'round_number']).zfill(2)

        scene_num = str(scene).zfill(2)
        position_num = Position[1::].zfill(2)
        well = dfsub["Well_id"][0]
        channels = reader.channel_names

        position = Position
        si = int(scene) - 1  # scene_index

        reader.set_scene(si)

        try:
            Tn = reader.dims["T"][0]  # choose last time point
            notcorrupt = True
        # except ValueError("This is a Corrupted Scene.") as e:
        except Exception as e:
            print(str(e), position)
            notcorrupt = False

        if notcorrupt:
            for T in range(Tn):

                for ci, c in enumerate(channels):
                    delayed_chunk = reader.get_image_dask_data("ZYX", T=T, C=ci)
                    imgstack = delayed_chunk.compute()

                    # this is where the alignment is performed
                    print(alignment_offset)
                    align_matrix = registration_utils.get_align_matrix(alignment_offset)
                    shift_to_center_matrix = registration_utils.get_shift_to_center_matrix(
                        imgstack.shape, final_shape
                    )
                    combo = shift_to_center_matrix @ align_matrix

                    # aligned image
                    processed_volume = affine_transform(
                        imgstack,
                        np.linalg.inv(combo),
                        output_shape=final_shape,
                        order=0,  # order = 0 means no interpolation...juust use nearest neighbor
                    )

                    cropped_maxz = np.max(
                        processed_volume, axis=0
                    )  # compute max z projection

                    # now save this image
                    output_dir = dfconfig["output_path"][0]

                    sep = os.sep
                    channel = c
                    channel_num = str(ci + 1).zfill(2)
                    tnum = str(T).zfill(3)
                    savedir = os.path.join(args.output_path, 'mip_exports' + args.test_save, f"{barcode}-export")


                    if not os.path.exists(savedir):
                        os.makedirs(savedir)
                        print("making", os.path.abspath(savedir))


                    fovid = f"{barcode}_R{round_num}_P{position_num}"
                    savename = (
                        f"{fovid}-mip-c{channel_num}_T{tnum}.tif"
                    )

                    savepath = f"{savedir}{sep}{savename}"
                    skio.imsave(
                        savepath, np.uint16(cropped_maxz), check_contrast=False
                    )
                    if (T == 0) & (ci == 0):
                        print(os.path.abspath(savepath))