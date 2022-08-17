import argparse
import os
from pathlib import Path
import re
import sys

from aicsimageio import AICSImage
import numpy as np
import pandas as pd
from scipy.ndimage import affine_transform
import skimage.io as skio
import yaml
from yaml.loader import SafeLoader

overwrite = True


def get_align_matrix(alignment_offset):
    align_matrix = np.eye(4)
    for i in range(len(alignment_offset)):
        align_matrix[i, 3] = alignment_offset[i] * -1
    align_matrix = np.int16(align_matrix)
    return align_matrix


def get_shift_to_center_matrix(img_shape, output_shape):
    # output_shape > img_shape should be true for all dimensions
    # and the difference divided by two needs to be a whole integer value

    shape_diff = np.asarray(output_shape) - np.asarray(img_shape)
    shift = shape_diff / 2

    shift_matrix = np.eye(4)
    for i in range(len(shift)):
        shift_matrix[i, 3] = shift[i]
    shift_matrix = np.int16(shift_matrix)
    return shift_matrix


def os_swap(x):
    out = "/" + ("/".join(x.split("\\"))).replace("//", "/")
    return out





# print(sys.argv)
# arg_list = [
#     (x.replace("--", ""), i)
#     for i, x in enumerate(list(sys.argv))
#     if bool(re.search("--", x))
# ]
# args_dict = {}

# for keyval in arg_list:
#     args_dict[keyval[0]] = sys.argv[keyval[1] + 1]
# print(args_dict)
# print()

# # args_dict['barcode'] = '5500000724'
# print(args_dict["barcode"])
# print()




parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path", type=str, required=True, help="output dir of all processing steps. This specifies where to find the yml_configs too"
)
parser.add_argument(
    "--barcode", type=str, required=True, help="specify barcode to analyze"
)




if __name__ == "__main__":
    args = parser.parse_args()


    # Open the file and load the file
    # Open the file and load the file
    barcode = args.barcode


    yaml_dir = os.path.join(args.output_path,"yml_configs")

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
    pickle_dir = output_dir + os.sep + "pickles"
    pickle_name = barcode + "cleanedup_match_pickle.pickle"
    pickle_path = pickle_dir + os.sep + pickle_name
    print("\n\n" + pickle_path + "\n\n")
    dfall = pd.read_pickle(pickle_path)


    output_dir = dfconfig["output_path"][0]
    align_pickle_dir = output_dir + os.sep + "alignment_pickles"
    align_pickle_name = barcode + "alignment_pickle.pickle"
    align_pickle_path = align_pickle_dir + os.sep + align_pickle_name
    dfalign = pd.read_pickle(align_pickle_path)
    dfalign.reset_index(inplace=True)
    dfalign.set_index(["key", "template_position"], inplace=True)


    dfall["parent_file"] = dfall["parent_file"].apply(lambda x: os_swap(x))


    template_position_list = dfall["template_position"].unique()
    keylist = dfall["key"].unique()
    # for Position in ['P2']:

    print(template_position_list)

    for Position in template_position_list: #go one position by position, since you need offsets per position
    # for Position in [
    #     "P6",
    #     "P3",
    #     "P12",
    # ]:  # go one position by position, since you need offsets per position
        print("POSITION = ", Position)
        
        #need to define the keylist for each position, since some positions may not be imaged every round
        keylist = dfall.set_index('template_position').loc[Position,'key'].unique()
        # testing_keylist = [x for x in keylist if "Time" not in x]
        # print(testing_keylist)
        # for ki, key in enumerate(testing_keylist):
        for ki,key in enumerate(keylist):
            print(key)

            dfr = dfall.set_index(["template_position", "key"])
            dfsub = dfr.loc[pd.IndexSlice[[Position], [key]], :]
            parent_file = dfsub["parent_file"][0]

            alignment_offset = dfalign.loc[
                pd.IndexSlice[key, Position], "alignment_offsets_xyz"
            ]
            final_shape = np.uint16(np.asarray(
                [
                    100,
                    1248 + 1248 / 3,
                    1848 + 1848 / 3,
                ]
            )
            )

            reader = AICSImage(parent_file)

            scene = dfsub["Scene"][0]

            # 3 get variables for file name
            # round_num0 = re.search("time|Round [0-9]+", parent_file, re.IGNORECASE).group(0)
            search_out =re.search("time|Round [0-9]+", parent_file, re.IGNORECASE)
            assert search_out is not None #necessary for passing mypy type errors
            round_num0 = search_out.group(0)

            round_num = round_num0.replace("Time", "0").replace("Round ", "").zfill(2)

            scene_num = str(scene).zfill(2)
            position_num = Position[1::].zfill(2)
            well = parent_file = dfsub["Well_id"][0]
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
                        align_matrix = get_align_matrix(alignment_offset)
                        shift_to_center_matrix = get_shift_to_center_matrix(
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
                        savedir = f"{output_dir}{sep}mip_exports{sep}{barcode}-export{sep}"

                        if not os.path.exists(savedir):
                            os.makedirs(savedir)
                            print("making", os.path.abspath(savedir))

                        file_name_stem = Path(dfsub["parent_file"].iloc[0]).stem

                        savename = (
                            f"{barcode}-{mag}-R{round_num}"
                            f"-Scene-{scene_num}-P{position_num}"
                            f"-{well}-maxproj_c{channel_num}_T{tnum}_ORG.tif"
                        )

                        savepath = f"{savedir}{sep}{savename}"
                        skio.imsave(savepath, np.uint16(cropped_maxz), check_contrast=False)
                        if (T == 0) & (ci == 0):
                            print(os.path.abspath(savepath))
