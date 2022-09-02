import argparse
import os
import skimage.exposure as skex
from aicsimageio import AICSImage
import matplotlib as plt
import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader
import registration_utils
import tifffile

overwrite = True

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dfall_csv_dir",
    required=True,
    type=str,
    help="csv file",
)

parser.add_argument(
    "--barcode", required=True, type=str, help="specify barcode to analyze"
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

parser.add_argument("--method", choices=['cross_cor', 'ORB', 'both'])


if __name__ == "__main__":
    args= parser.parse_args()
    barcode = args.barcode
    Position = args.position

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
                # print(round_dict)
                # reader = AICSImage(round_dict['path'])
                # channels = reader.channel_names
                # print(Path(round_dict['path']).name)
                # print(data['Data'][0])
                # print()
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
    print(f"pickle path is {pickle_path}")
    dfall = pd.read_pickle(pickle_path)
    #dfkeeplist = []
    print(f"position {Position}")
    keeplist = []
    # need to define the keylist for each position, since some positions may not be imaged every round
    keylist = dfall.set_index("template_position").loc[Position, "key"].unique()
    print(f"keylist is {keylist}")


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
            #TODO: There is an issue where everything in z is not being loaded for the very first image...... is this just inconsistent imageing or is this an AICSIMAGEIO issue???
            print(f"imgstack is of shape {np.shape(imgstack)}")
            # print(imgstack.shape)
            keeplist.append((align_channel, key, imgstack))

    # now align images
    # this workflow does not do camera alignment
    # this workflow only does tranlsation (no scaling and no rotation)

    dfimg = pd.DataFrame(keeplist, columns=["align_channel", "key", "img"])
    dfimg.set_index("key", inplace=True)
    # dfimg.loc['Round 1'][['align_channel']]

    keylist = dfimg.index.values

    img_list = []
    for key in keylist:
        align_channel = dfimg.loc[key, "align_channel"]
        imgstack = dfimg.loc[key, "img"]
        img_list.append(imgstack.copy())
    print(len(img_list))
    print(f"img list is of type {type(img_list)}")
    print(f"shape of img list is {np.shape(img_list)}")
    print(f"first image in list is of shape {np.shape(img_list[1])}")

    # tifffile.imwrite("img_list_0.tiff", img_list[1])
    # tifffile.imwrite("img_list_1.tiff", img_list[2])



    reference_round_key = "Round 1"
    refimg = dfimg.loc[reference_round_key, "img"]
    alignment_offsets_xyz_list = registration_utils.find_xyz_offset_relative_to_ref(
        img_list, refimg=refimg, ploton=False, verbose=False
    )
    source_cropping=100
    if args.method == "ORB" or args.method=="both":
        print(f"alignment offset xyz is {alignment_offsets_xyz_list}")

        #target_img_padded = np.pad(refimg, ((0, 0), (50, 50), (50, 50)), mode='constant')
        # tifffile.imwrite("refimg.tiff", refimg)
        alignment_offsets_xyz_list_ORB= []


        for i in range(len(img_list)):
            try:
                print(f"looking at {i}")
                new_source_img = img_list[i]
                source_img_FOV_cropped = new_source_img[:, source_cropping:np.shape(new_source_img)[1]-source_cropping, source_cropping:np.shape(new_source_img)[2]-source_cropping]

                #source_img_padded = np.pad(source_img_FOV, ((0, 0), (50, 50), (50, 50)), mode='constant')
                print(f"shape of ref img is {np.shape(refimg)}")
                print(f"shape of source image is {np.shape(source_img_FOV_cropped)}")
                # tifffile.imwrite(f"source_img_FOV_{i}_padded.tiff", source_img_FOV_cropped)

                #ref_img_cropped = refimg[:,100:np.shape(refimg)[1]-100, 100:np.shape(refimg)[2]-100]
                # the indexing here is z,y,x -- check this
                final_y_offset, final_x_offset = registration_utils.perform_alignment(source_img_FOV_cropped, refimg, smaller_fov_modality="source", scale_factor_xy=1, scale_factor_z=1, source_alignment_channel=0, target_alignment_channel=0, source_output_channel=[0], target_output_channel=[0], prealign_z=True, denoise_z=True, use_refinement=False, save_composite=False)
                print(final_y_offset, final_x_offset)
                # match_list = align_helper.return_aligned_img_list_new(img_list,offset_list)
                print("for first img in new_source_img, x_offset is", final_x_offset)
                print("for first img in new_source_img, y_offset is", final_y_offset)
                alignment_offsets_xyz_list_ORB.append((final_x_offset, final_y_offset))
            except:
                alignment_offsets_xyz_list_ORB.append((0,0))

        final_version_offsets_zyx=[]
        # This just copies over the z offset from the cross correlation method
        for k in range(len(alignment_offsets_xyz_list_ORB)):
            dat=alignment_offsets_xyz_list_ORB[k]
            final_version_offsets_zyx.append([alignment_offsets_xyz_list[k][0], source_cropping-dat[0], source_cropping-dat[1]])
        
        print(f"final version offset xyz is {final_version_offsets_zyx}")
        print("outputing")
        dfimg["alignment_offsets_xyz_ORB"] = final_version_offsets_zyx
        dfimg["alignment_offsets_xyz_cross_corr"] = alignment_offsets_xyz_list

    elif args.method=="cross_cor":
        dfimg["alignment_offsets_xyz_cross_corr"] = alignment_offsets_xyz_list
        dfimg["alignment_offsets_xyz_ORB"] = [np.NaN for f in alignment_offsets_xyz_list]
        


    print("dfimg.shape[0] is {}".format(dfimg.shape[0]))
    dfimg["template_position"] = [Position] * dfimg.shape[0] # This is what?
    dfout_p = dfimg[["template_position", "align_channel", "alignment_offsets_xyz_ORB", "alignment_offsets_xyz_cross_corr"]]
    #dfkeeplist.append(dfout_p)

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
    print("succesfully computed alignment parameters and saved")





