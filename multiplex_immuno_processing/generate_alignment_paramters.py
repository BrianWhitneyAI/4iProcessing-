import argparse
import os
from aicsimageio import AICSImage
import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader
import registration_utils

overwrite = True

parser = argparse.ArgumentParser()


parser.add_argument(
    "--barcode",
    required=True,
    type=str,
    help="specify barcode to analyze"
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


parser.add_argument(
    "--test_save",
    type=str,
    required=False,
    help="modify output directory by appending 'test' (default='')",
    default = '',
)

parser.add_argument("--method",
            choices=['cross_cor', 'ORB', 'both'])


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

        channels = reader.channel_names

        align_channel = dfsub["align_channel"][0]

        position = Position
        scene = dfsub["Scene"][0]

        align_channel_index = [
            xi for xi, x in enumerate(channels) if x == align_channel
        ][0]

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
            keeplist.append((align_channel, key, imgstack))

    # now align images
    # this workflow does not do camera alignment
    # this workflow only does tranlsation (no scaling and no rotation)

    dfimg = pd.DataFrame(keeplist, columns=["align_channel", "key", "img"])
    dfimg.set_index("key", inplace=True)

    keylist = dfimg.index.values

    
    reference_round_key = "Round 1"
    refimg = dfimg.loc[reference_round_key, "img"]
    

    alignment_offsets_xyz_list = []
    failed_list = []
    final_version_offsets_zyx_list=[]
    cross_corr_offset_zyx_list =[]
    method_list = []

    # iterate through each round and append the outcomes to the lists above
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
        

        cross_corr_offset_zyx_list.append(meanoffset)
        
        
        if args.method == "ORB" or args.method=="both":
            source_cropping=100
            
            try:
                new_source_img = imgstack
                source_img_FOV_cropped = new_source_img[:,
                                                        source_cropping:np.shape(new_source_img)[1]-source_cropping,
                                                        source_cropping:np.shape(new_source_img)[2]-source_cropping]

                
                print(f"shape of ref img is {np.shape(refimg)}")
                print(f"shape of source image is {np.shape(source_img_FOV_cropped)}")
                
                #ref_img_cropped = refimg[:,100:np.shape(refimg)[1]-100, 100:np.shape(refimg)[2]-100]
                # the indexing here is z,y,x -- check this
                final_y_offset, final_x_offset = registration_utils.perform_alignment(source_img_FOV_cropped,
                                                                                        refimg,
                                                                                        smaller_fov_modality="source",
                                                                                        scale_factor_xy=1,
                                                                                        scale_factor_z=1,
                                                                                        source_alignment_channel=0,
                                                                                        target_alignment_channel=0,
                                                                                        source_output_channel=[0],
                                                                                        target_output_channel=[0],
                                                                                        prealign_z=True,
                                                                                        denoise_z=True,
                                                                                        use_refinement=False,
                                                                                        save_composite=False)

                # need to adjust offset for the source_cropping performed above
                final_y_offset = source_cropping - final_y_offset
                final_x_offset = source_cropping - final_x_offset
                
                print("for first img in new_source_img, x_offset is", final_x_offset)
                print("for first img in new_source_img, y_offset is", final_y_offset)

            except:
                final_x_offset = 0
                final_y_offset = 0
                failed=True
            
            alignment_offsets_xyz_ORB = (final_x_offset, final_y_offset)

            #now this method collects the alignment in Z from the cross correlation method
            final_version_offsets_zyx = [meanoffset[0],
                                        alignment_offsets_xyz_ORB[0],
                                        alignment_offsets_xyz_ORB[1],
                                        ]

            method = 'ORB'
            
            
            
            print(f"final version offset xyz is {final_version_offsets_zyx}")
            print("outputing")


        elif args.method=="cross_cor":
            final_version_offsets_zyx = meanoffset
            method = 'cross_cor'
            # failed is determined above

            

        # record outcomes
        final_version_offsets_zyx_list.append(final_version_offsets_zyx)
        method_list.append(method)
        failed_list.append(failed)



    dfimg['alignment_offsets_xyz'] = final_version_offsets_zyx_list
    dfimg['method'] = method_list
    dfimg['failed'] = failed_list
    dfimg['alignment_offsets_xyz_cross_cor'] = cross_corr_offset_zyx_list
            


    print("dfimg.shape[0] is {}".format(dfimg.shape[0]))
    dfimg["template_position"] = [Position] * dfimg.shape[0] # This is what?
    dfout_p = dfimg[["template_position", "align_channel", "alignment_offsets_xyz", "alignment_offsets_xyz_cross_cor","method"]]
    #dfkeeplist.append(dfout_p)


    output_dir = dfconfig["output_path"][0]
    pickle_dir = output_dir + os.sep + "alignment_pickles_each" + args.test_save
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





