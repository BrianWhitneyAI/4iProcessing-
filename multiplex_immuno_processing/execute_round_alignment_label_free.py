import argparse
from cProfile import label
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

parser.add_argument(
    "--method", choices=['cross_cor', 'ORB', 'merged'])


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
    align_csv_dir = output_dir + os.sep + "alignment_csvs_each_" + args.method
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


        key = '20X_Timelapse'

        print(key)


        label_free_dir = os_swap(r"\\allen\aics\microscopy\EMTImmunostainingResults\4iTimelapse_UnalignedLabelFreeImages\UnalignedLabelFreeImages")
        

        dfr = dfall.set_index(["template_position", "key"])
        dfsub = dfr.loc[pd.IndexSlice[[Position], [key]], :]
        parent_file = dfsub["parent_file"][0]

        if args.method == "merged":
            print("using merged")
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

        
        zpad = 20
        xypad = 300
        final_shape = np.uint16(
            np.asarray(
                [
                    45 + zpad*2,
                    1248 + xypad*2,
                    1848 + xypad*2,
                ]
            )
        )

        reader = AICSImage(parent_file)

        scene = dfsub["Scene"][0]

        # # 3 get variables for file name
        # # round_num0 = re.search("time|Round [0-9]+", parent_file, re.IGNORECASE).group(0)
        # search_out = re.search("time|Round [0-9]+", parent_file, re.IGNORECASE)
        # assert search_out is not None  # necessary for passing mypy type errors
        # round_num0 = search_out.group(0)

        # round_num = round_num0.replace("Time", "0").replace("Round ", "").zfill(2)
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

            # define the label-free channel as an extra channel
                ci = len(channels) +1 
                sep = os.sep
                channel = 'label_Free'
                channel_num = str(ci + 1).zfill(2)
                tnum = str(T).zfill(3)
                savedir = (
                    f"{output_dir}{sep}mip_exportstest_v3_merged{sep}{barcode}-export{sep}"
                )


                fovid = f"{barcode}_R{round_num}_P{position_num}"
                savename = (
                    f"{fovid}-mip-c{channel_num}_T{tnum}.tif"
                )
                savepath = os.path.abspath(f"{savedir}{sep}{savename}")
                


                glob_label_free_name = f"{barcode}-20x-R{round_num}-Scene-{scene_num}-P{str(int(position_num))}*-maxproj_c02_T{tnum}_ORG_ProbabilitiesUnaligned.tif"
                glob_path = label_free_dir + os.sep + glob_label_free_name
                from glob import glob
                print("glob path is")
                print(glob_path)
                globlist = glob(glob_path)
                load_path = globlist[0]
                print(f"load path is {load_path}")
                if len(globlist)>1:
                    print('multiplematchesfound')
                print(os.path.abspath(load_path))
                #print(r"\\allen\aics\microscopy\EMTImmunostainingResults\4iTimelapse_UnalignedLabelFreeImages\UnalignedLabelFreeImages\5500000728-20x-R00-Scene-39-P39-C09-maxproj_c02_T016_ORG_ProbabilitiesUnaligned.tif")
                # //allen/aics/microscopy/EMTImmunostainingResults/4iTimelapse_UnalignedLabelFreeImages/UnalignedLabelFreeImages/5500000724-20x-R00-Scene-34-P34-F9-maxproj_c02_T000_ORG_ProbabilitiesUnaligned.tif
                # \\allen\aics\microscopy\EMTImmunostainingResults\4iTimelapse_UnalignedLabelFreeImages\UnalignedLabelFreeImages\5500000728-20x-R00-Scene-39-P39-C09-maxproj_c02_T016_ORG_ProbabilitiesUnaligned.tif

                lreader = AICSImage(load_path)
                #if not os.path.exists(savepath):

                    # print(savepath)
                    # print(os.path.exists(savepath))

                    
                    # print(fovid)
                    # print(scene,si,T,ci)
                    # print(dfsub)
                    # print(reader.dims)
                    # print(reader.current_scene)
                    # print(reader.current_scene_index)
                    # print(parent_file)
                    # print(dfsub["parent_file"][0])

                
                imgstack = lreader.get_image_data("ZYX")
                print(imgstack.shape)
                

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

                

                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                    print("making", os.path.abspath(savedir))

                file_name_stem = Path(dfsub["parent_file"].iloc[0]).stem

                
                skio.imsave(
                    savepath, np.uint16(cropped_maxz), check_contrast=False
                )
                if (T == 0) & (ci == 0):
                    print(os.path.abspath(savepath))
                # else:
                #     print(f"already processed{savename}")
