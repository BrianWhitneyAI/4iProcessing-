import argparse
import os

from aicsimageio import AICSImage
import matplotlib as plt
import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader
import registration_utils

overwrite = True

# this code computes the alignment parameters to align each scene across the multiple rounds of imaging
# should take a barcode as an argument,
# then this code reads that dataframe pickle for that given barcode
# then it uses the dataframe to determine the reference channel to be used for each of the rounds for alignment
# then it loads all the reference channel images for each round
# then it runs the an alignment algorithm to align all of the rounds
# all positions should be aligned to round 1 (that will be the reference round)
# this will enable all the processing to be run in stages later on as new data is acquire

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


if __name__ == "__main__":
    args = parser.parse_args()

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
    pickle_dir = output_dir + os.sep + "pickles"
    pickle_name = barcode + "cleanedup_match_pickle.pickle"
    pickle_path = pickle_dir + os.sep + pickle_name
    print("\n\n" + pickle_path + "\n\n")
    print(os.path.exists(pickle_path))
    # dfall = pd.read_pickle(pickle_path)

    pickle_name = barcode + "cleanedup_match_csv.csv"
    pickle_path = pickle_dir + os.sep + pickle_name
    print("\n\n" + pickle_path + "\n\n")
    print(os.path.exists(pickle_path))
    dfall = pd.read_csv(pickle_path)

    def os_swap(x):
        out = "/" + ("/".join(x.split("\\"))).replace("//", "/")
        return out

    dfall["parent_file"] = dfall["parent_file"].apply(lambda x: os_swap(x))

    template_position_list = dfall["template_position"].unique()
    # keylist = mag_dict.keys()
    keylist = dfall["key"].unique()
    # for Position in ['P2']:

    print(template_position_list)
    dfkeeplist = []
    # go one position by position, since you need offsets per position
    # template_position_list = [x for x in template_position_list if 'P9' == x]
    for Position in template_position_list:
        # for Position in ['P6', 'P3', 'P12']: #go one position by position, since you need offsets per position
        print("POSITION = ", Position)
        keeplist = []

        # need to define the keylist for each position, since some positions may not be imaged every round
        keylist = dfall.set_index("template_position").loc[Position, "key"].unique()

        for key in keylist:

            dfr = dfall.set_index(["template_position", "key"])
            dfsub = dfr.loc[pd.IndexSlice[[Position], [key]], :]
            parent_file = dfsub["parent_file"][0]

            # print(key,Path(parent_file).name,Position)

            reader = AICSImage(parent_file)

            # channel_dict = get_channels(czi)
            channels = reader.channel_names
            # print('channels found = ', channels)

            align_channel = dfsub["align_channel"][0]
            # print('align_channel found = ', align_channel)

            position = Position
            scene = dfsub["Scene"][0]

            align_channel_index = [
                xi for xi, x in enumerate(channels) if x == align_channel
            ][0]
            # print(position,' - S' + str(scene).zfill(3) )
            si = int(scene) - 1  # scene_index

            reader.set_scene(si)
            # if scene has no image data (i.e. is corrupted) then reader.dims will error
            try:
                T = reader.dims["T"][0] - 1  # choose last time point
                notcorrupt = True
            # except ValueError("Something has gone wrong. This is a Corrupted Scene.") as e:
            except Exception as e:
                print(str(e))
                notcorrupt = False

            if notcorrupt:
                align_chan = dfall.groupby("key").agg("first").loc[key, "align_channel"]
                delayed_chunk = reader.get_image_dask_data(
                    "ZYX", T=T, C=align_channel_index
                )
                imgstack = delayed_chunk.compute()
                # print(imgstack.shape)
                keeplist.append((align_channel, key, imgstack))

        # now align images
        # this workflow does not do camera alignment
        # this workflow only does tranlsation (no scaling and no rotation)

        dfimg = pd.DataFrame(keeplist, columns=["align_channel", "key", "img"])
        dfimg.set_index("key", inplace=True)
        # dfimg.loc['Round 1'][['align_channel']]


        reference_round_key = "Round 1"
        refimg = dfimg.loc[reference_round_key, "img"]
        # alignment_offsets_xyz_list = registration_utils.find_xyz_offset_relative_to_ref(
        #     img_list, refimg=refimg, ploton=False, verbose=False
        # )


        keylist = dfimg.index.values

        alignment_offsets_xyz_list = []
        failed_list = []
        for key in keylist:
            align_channel = dfimg.loc[key, "align_channel"]
            imgstack = dfimg.loc[key, "img"]

            # try to align with current code
            try:
                (_, _, meanoffset, _,) = registration_utils.find_xyz_offset(
                    refimg.copy(), imgstack.copy(), ploton=False, verbose=False,
                )
                failed = False
            except Exception as e:
                print(str(e))
                failed = True
                print(f"{barcode}-{Position}-{key} has failed alignment")
                meanoffset = [0,0,0]
            
            # record outcomes
            alignment_offsets_xyz_list.append(meanoffset)
            failed_list.append(failed)


    
        
        

        # match_list = align_helper.return_aligned_img_list_new(img_list,offset_list)
        print("alignment_offsets_xyz_list\n", alignment_offsets_xyz_list)
        # unmatch_list = align_helper.return_aligned_img_list_new(img_list,np.asarray(offset_list)*0)

        dfimg["alignment_offsets_xyz"] = alignment_offsets_xyz_list
        dfimg["template_position"] = [Position] * dfimg.shape[0]
        dfimg["failed"] = failed_list
        dfout_p = dfimg[["template_position", "align_channel", "alignment_offsets_xyz", "failed"]]
        dfkeeplist.append(dfout_p)

        output_dir = dfconfig["output_path"][0]
        pickle_dir = output_dir + os.sep + "alignment_pickles_each"
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        pickle_name = f"{barcode}-{Position}-alignment_pickle_each.pickle"
        pickle_path = pickle_dir + os.sep + pickle_name
        print("\n\n" + pickle_path + "\n\n")
        dfout_p.to_pickle(os.path.abspath(pickle_path))

        # out_csv_path = pickle_path.replace('_pickle','_csv').replace('.pickle','.csv')
        csv_name = pickle_name.replace("_pickle", "_csv").replace(".pickle", ".csv")
        out_csv_path = pickle_dir + os.sep + csv_name
        dfout_p.to_csv(os.path.abspath(out_csv_path))

    # now load up the full images (including the full timelapse)

    dfout = pd.concat(dfkeeplist)

    output_dir = dfconfig["output_path"][0]
    pickle_dir = output_dir + os.sep + "alignment_pickles"
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
    pickle_name = barcode + "alignment_pickle.pickle"
    pickle_path = pickle_dir + os.sep + pickle_name
    print("\n\n" + pickle_path + "\n\n")
    dfout.to_pickle(os.path.abspath(pickle_path))

    # out_csv_path = pickle_path.replace('_pickle','_csv').replace('.pickle','.csv')
    csv_name = pickle_name.replace("_pickle", "_csv").replace(".pickle", ".csv")
    out_csv_path = pickle_dir + os.sep + csv_name
    dfout.to_csv(os.path.abspath(out_csv_path))
