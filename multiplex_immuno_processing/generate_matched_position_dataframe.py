import argparse
from collections import namedtuple
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader

pd.set_option("display.max_columns", None)
ploton = False

# TODO: parse argument to decide which barcode to run?

"""
0. find the yaml files for each barcode (currently finds all barcodes)

1. retrieve positions/scenes/coordinates from experiment file for acquired positions

2. remove scenes that are marked for removal

3. CHECK1: for each experiment --  determine which FOVs overlap and remove those overlapping FOVs.
    a. in near future it would be good to keep the FOV that was acquired first. Actually this is a bad idea.
    b. it is a bad idea because the overlap will cause bleaching in the images acquired with the next modality!!!!!
    c. At this point all the keepable FOVs are kept and all the incorrect/overlapping/problematic FOVs are removed
    d. FUTURE: dont remove these positions...instead they should be flagged so the user knows that there was an issue.

4. Now set one imaging round to be the fixed "template"/"temp" dataframe/positionlist.
    a. for each FOV in this template imaging round, look for FOVs in the subsequent
       imaging round that overlap to identify matched sets.
    b. the same thing is said again in line 5.

5. The next step is to match positions of the subsequent imaging rounds to the template dataframe/positionlist.
    a. align all positions to one set of imaging that will be set as the template
    b. all other datasets will be aligned to this and indexed with the positon number of the template dataset
    c. TODO: check on how to handle this. right now if a round had an extra position that didn't match to the timelapse
        (or template round) then that would get tossed without any notification or comment.Data is just gone because
        its assumed to be superfluoes.

6. Then after the matches are generated, assemble into a dataframe and:
    a. export the csv or pickle that defines the matched positions

The idea is that this dataframe contains all of the information needed for calling any other processing step.

TODO: decide if we want any contact sheet-style outputs from this code that help the user know what is being done.

    IDEA: create max projection tiled output that looks to see if the same FOV was
          indeed imaged for all "putative" matched positions.
"""


def create_rectangle(xyz, imgsize_um):
    """
    credit: https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles
    """

    Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")
    rect = Rectangle(
        xyz[0] - imgsize_um[1] / 2,
        xyz[1] - imgsize_um[0] / 2,
        xyz[0] + imgsize_um[1] / 2,
        xyz[1] + imgsize_um[0] / 2,
    )
    return rect


def intersection_area(a, b):  # returns None if rectangles don't intersect
    """ "
    requires a ==> Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    credit: https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles
    """
    area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin)
    area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin)
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy / np.min((area_a, area_b))
    else:
        return 0


# first aggregate all files into a dataframe with metadata
def plot_position_rectangles(dfforplot, fs=12, figsize=(5, 5)):
    dfkeep = dfforplot.copy()
    dfkeep.reset_index(inplace=True)

    # now plot the overlap of different files
    # make a scatter plot of positions
    # npt = 1  # number of 20x positions to examine
    colorlist = ["k", "r", "c", "y", "g"]
    coloriter = cycle(colorlist)
    remainder = np.ceil(len(dfkeep.parent_file.unique())/len(colorlist))
    if remainder>1:
        coloriter = []
        for remainder_int in remainder:
            coloriter.extend(colorlist)
    else:
        coloriter=colorlist

    color_dict = {
        x: coloriter[xi] for xi, x in enumerate(dfkeep.parent_file.unique())
    }
    print(color_dict)
    for well, df0 in dfkeep.groupby("Well_id"):
        plt.figure(figsize=figsize)

        ncols = len(df0["parent_file"].unique()) + 1
        fig, ax = plt.subplots(
            nrows=1,
            ncols=ncols,
            figsize=(figsize[0] * ncols, figsize[1]),
            sharex=True,
            sharey=True,
        )
        axlist = ax.reshape(
            -1,
        )

        for i, (img_label, df1) in enumerate(df0.groupby("parent_file")):
            X = df1["X"].to_numpy()
            Y = df1["Y"].to_numpy()
            PositionList = df1["Position"].tolist()

            imgsize_um = df1["imgsize_um"].to_numpy()[0]
            w = imgsize_um[1]
            h = imgsize_um[0]

            for ii in [i, ncols - 1]:
                plt.sca(axlist[ii])
                for xx, y, pos in zip(X, Y, PositionList):
                    plt.fill_between(
                        x=(xx - w / 2, xx + w / 2),
                        y1=(y - h / 2, y - h / 2),
                        y2=(y + h / 2, y + h / 2),
                        facecolor=color_dict[img_label],
                        alpha=0.2,
                        edgecolor=(0, 0, 0, 0),
                    )
                    plt.text(xx, y, pos, fontsize=fs)
                plt.title(well + "\n" + Path(img_label).stem, fontsize=fs)

                # plt.title(well,fontsize=fs)
                plt.axis("square")
                # xlim = plt.xlim()
                # ylim = plt.ylim()
                # lims = [np.min([xlim[0],ylim[0]]),
                # np.max([xlim[1],ylim[1]])]
                # plt.xlim(lims)
                # plt.ylim(lims)
        plt.show()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="output dir of all processing steps. This specifies where to find the yml_configs too",
)


if __name__ == "__main__":
    args = parser.parse_args()

    # load the yaml config files and populate a dataframe with config info
    yaml_dir = os.path.join(args.output_path, "yml_configs")
    yaml_list = [x for x in os.listdir(yaml_dir) if "_confirmed" in x]
    dfconfiglist = []
    for y in yaml_list:
        print(y)
        yml_path = yaml_dir + os.sep + y
        with open(yml_path) as f:
            data = yaml.load(f, Loader=SafeLoader)
            for round_dict in data["Data"]:
                dfconfigsub = pd.DataFrame(
                    round_dict.values(), index=round_dict.keys()
                ).T
                dfconfigsub["barcode"] = data["barcode"]
                dfconfigsub["scope"] = data["scope"]
                dfconfigsub["output_path"] = data["output_path"]
                dfconfiglist.append(dfconfigsub)

    dfconfig = pd.concat(dfconfiglist)

    # dfconfig = dfconfig[dfconfig['barcode']=='5500000725'] #for testing

    dfconfig.set_index(["barcode", "round"], inplace=True)

    # now open the metadata pickle
    # (specifically metadata about position, XYZ coordinates and FOV size)
    # barcode corresponds to a given plate
    # round corresponds to a given round of imaging of that plate
    for barcode, dfcb in dfconfig.groupby(["barcode"]):

        output_dir = dfconfig["output_path"][0]
        metadata_pickle_dir = output_dir + os.sep + "pickles"
        metadata_pickle_name = barcode + "metadata_pickle.pickle"
        metadata_pickle_path = metadata_pickle_dir + os.sep + metadata_pickle_name

        dfmeta = pd.read_pickle(metadata_pickle_path)

        PositionUniqueList = [
            f"{str(x)}-{y}-{str(Path(z).name)}"
            for x, y, z in zip(
                dfmeta["Position"].tolist(),
                dfmeta["key"].tolist(),
                dfmeta["parent_file"].tolist(),
            )
        ]
        dfmeta["PositionUnique"] = PositionUniqueList
        dfmetaog = dfmeta.copy()

        # this has all metadata for all rounds of image data for one given barcode

        # important columns are:
        # ['original_file',
        #  'file',
        #  'parent_file',
        #  'imgsize_pixels',
        #  'ImagePixelDistances',
        #  'totalmagnification',
        #  'channel_dict',
        #  'pixelSizes',
        #  'imgsize_um',
        #  'PlateAnchorPoint',
        #  'PlateReferencePoint',
        #  'X',
        #  'Y',
        #  'Z',
        #  'IsUsedForAcquisition',
        #  'Position',
        #  'Position_num',
        #  'Scene',
        #  'Well_id',
        #  'X_original',
        #  'X_adjusted',
        #  'Y_original',
        #  'Y_adjusted',
        #  'align_channel',
        #  'barcode',]

        # TODO: define why use anchor point in zen_position_helper
        # dfall[['X','X_original','X_adjusted','PlateReferencePoint','PlateAnchorPoint']]

        if ploton:
            plot_position_rectangles(dfmeta, fs=18, figsize=(10, 10))

        # now remove the scenes specified in the yaml config above
        original_file_AND_scenes_to_toss_list = []
        # for key,value in mag_dict.items():
        for round, dfcbr in dfcb.groupby(["round"]):

            # convert the string in the yaml file to a list of lists
            scenes_to_toss_list = [
                eval("[" + x + "]") for x in dfcbr["scenes_to_toss"].tolist()
            ]

            # now iterate through all original files in this round of imaging
            # since each original file can have its own list of scenes to toss
            # within this loop create a list of tuples that will be kept as lables to pass
            # into a dataframe's drop argument.
            original_file_list = dfcbr.path.tolist()
            for oi, original_file in enumerate(original_file_list):
                scenes_to_toss = scenes_to_toss_list[oi]
                print(Path(original_file).name, scenes_to_toss)
                original_file_AND_scenes_to_toss_list.extend(
                    [(original_file, scene) for scene in scenes_to_toss]
                )

        #########################################################
        # now flag all scenes that should be tossed

        # set the index to match the collected information in the list of tuples above
        # should be "original_file" and "Scene"
        dfmeta.reset_index(inplace=True)
        dfmeta.set_index(
            ["original_file", "Scene"],
            inplace=True,
        )

        dfmeta["flag-scene_to_toss"] = [False] * dfmeta.shape[0]
        for original_file_AND_scenes_to_toss in original_file_AND_scenes_to_toss_list:
            try:
                dfmeta.loc[
                    pd.IndexSlice[original_file_AND_scenes_to_toss],
                    "flag-scene_to_toss",
                ] = True
            except Exception as e:
                print(str(e))  # this excepts because some scenes are not present????
                print("why", original_file_AND_scenes_to_toss)
                pass

        # the last step is to remove all extra scenes that were added
        # (as scenes many scenes marked for removal wont be present in the
        # metadata and so could be added to the dataframe using the loc method above)
        dfmeta = dfmeta[dfmeta.isna()["parent_file"] is False]
        dfmeta
        #########################################################

        # # now drop the identified "scenes to toss" by passing in the list of tuples
        # dfmeta.drop(
        #     labels=original_file_AND_scenes_to_toss_list, inplace=True, errors="ignore"
        # )

        # find and record overlapping FOVs (FOVs that overlap within the same round of imaging)
        # overlapping FOVs will cause bleaching, so it is good to remove these.
        # positions_to_remove_list = []

        dfoverlaplist = []
        for key, df in dfmeta.groupby("key"):

            print(key)
            dfl = []
            for i, ((pos, pf, pu), dftemp_pos) in enumerate(
                df.groupby(["Position", "parent_file", "PositionUnique"])
            ):

                xyz = dftemp_pos[["X", "Y", "Z"]].to_numpy()[0]
                imgsize_um = dftemp_pos["imgsize_um"].to_numpy()[0]
                template_rectangle = create_rectangle(xyz, imgsize_um)

                for k, ((pos2, pf2, pu2), dfmove_pos) in enumerate(
                    df.groupby(["Position", "parent_file", "PositionUnique"])
                ):
                    xyz2 = dfmove_pos[["X", "Y", "Z"]].to_numpy()[0]
                    imgsize_um2 = dfmove_pos["imgsize_um"].to_numpy()[0]
                    move_rectangle = create_rectangle(xyz2, imgsize_um2)

                    overlap = intersection_area(template_rectangle, move_rectangle)

                    # must overlap, and must not be the same position name (unless its from different parent files)

                    # if the positions overlap AND (they don't have the same position name OR
                    # they are from a different parent file) then mark them for removal.
                    if (overlap > 0) & (pu != pu2):
                        feats = {}
                        #             print(pos,pos2,overlap)
                        feats["template_position"] = pos
                        feats["move_position"] = pos2
                        feats["template_position_unique"] = pu
                        feats["move_position_unique"] = pu2
                        feats["overlap"] = overlap
                        feats["template_parent_file"] = pf
                        feats["move_parent_file"] = pf2
                        feats["key"] = key
                        dfl.append(
                            pd.DataFrame(data=feats.values(), index=feats.keys()).T
                        )

            if len(dfl) >= 1:  # needed to account for if no overlap occurs
                dfoverlap_self = pd.concat(dfl)
                dfoverlaplist.append(dfoverlap_self)

        #########################################################
        # now flag all positions that have same-round overlaps
        # and record which positions overlap
        dfmeta["flag-overlaps_with_another_position_in_same_round"] = [
            False
        ] * dfmeta.shape[0]
        dfmeta["overlapping_positions_within_same_round"] = [
            "No overlap"
        ] * dfmeta.shape[0]
        if len(dfoverlaplist) >= 1:  # needed to account for if no overlap occurs
            dfoverlapall = pd.concat(dfoverlaplist)
            # now record all the overlapping positions
            dfmeta.reset_index(inplace=True)
            dfmeta.set_index(["key", "PositionUnique"], inplace=True)
            for index_item, dfoverlapallg in dfoverlapall.groupby(
                ["key", "template_position_unique"]
            ):
                # dfmeta.loc[pd.IndexSlice[index_item],'overlapping_positions_within_same_round'] =
                # str([[str(y) for y in x] for x in dfoverlapallg[['move_position','move_parent_file']].values])
                # #need string operation on list comprehension to get multiple items at single location in dataframe
                # dfmeta.loc[pd.IndexSlice[index_item],'overlapping_positions_within_same_round'] =
                # str([[str(y) for y in x] for x in dfoverlapallg['move_position_unique'].tolist()])
                # #need string operation on list comprehension to get multiple items at single location in dataframe
                dfmeta.loc[
                    pd.IndexSlice[index_item], "overlapping_positions_within_same_round"
                ] = str(
                    [str(x) for x in dfoverlapallg["move_position_unique"].tolist()]
                )  # need string operation on list comprehension to get multiple items at single location in dataframe
                dfmeta.loc[
                    pd.IndexSlice[index_item],
                    "flag-overlaps_with_another_position_in_same_round",
                ] = True

        print(
            dfmeta[dfmeta["flag-overlaps_with_another_position_in_same_round"]][
                "overlapping_positions_within_same_round"
            ]
        )
        #########################################################
        # dfmeta.drop(labels=positions_to_remove_list, inplace=True, errors="ignore")
        print("dfmeta.shape", dfmeta.shape)

        # # ploton=True
        # if ploton:
        #     plot_position_rectangles(dfmeta)

        # find the same FOV across multiple rounds of imaging by finding FOVs that overlap
        # from two different rounds.
        # this is done by looking at overlap of the FOV coordinates in xyz across rounds.

        # define the list to start with Round 1
        ukeys = dfmeta.reset_index()["key"].unique()
        # create list with timelapse as first (unique should sort the numbering for the other rounds)
        keylist0 = [x for x in ukeys if "Time" in x] + [
            x for x in ukeys if "Time" not in x
        ]

        # then set the first entry to be round 1
        keylist = ["Round 1"] + [x for x in keylist0 if "Round 1" != x]
        print(keylist)

        # keeplist = []
        dflall = []

        dfmeta.reset_index(inplace=True)
        dfmeta.set_index(["key", "PositionUnique"], inplace=True)

        # # extra junk to use for testing
        # dfmeta.loc[('20X_Timelapse', 'P1-5500000725_20X_Timelapse-01.czi'),'X'] = 1
        # dfmeta.loc[('20X_Timelapse', 'P4-5500000725_20X_Timelapse-01.czi'),['X','Y','Z']] =
        # dfmeta.loc[('20X_Timelapse', 'P3-5500000725_20X_Timelapse-01.czi'),['X','Y','Z']].values
        # display(dfmeta.loc[pd.IndexSlice['20X_Timelapse',['P1-5500000725_20X_Timelapse-01.czi',
        #                                                   'P2-5500000725_20X_Timelapse-01.czi',
        #                                                   'P3-5500000725_20X_Timelapse-01.czi',
        #                                                   'P4-5500000725_20X_Timelapse-01.czi',]],:])

        # dye
        print("template round is = ", keylist[0])
        for ki in range(0, len(keylist)):
            dfmeta.reset_index(inplace=True)
            dfmeta.set_index(["key"], inplace=True)
            template_slice = pd.IndexSlice[
                keylist[0]
            ]  # should be first round or time lapse...defined up above

            dftemplate = dfmeta.loc[
                template_slice, :
            ]  # template set to which other sets are matched to.

            move_slice = pd.IndexSlice[keylist[ki]]
            dfmove = dfmeta.loc[move_slice, :]  # set to be matched/"moved" to template

            print(ki, move_slice)

            # find and record overlapping FOVs
            print("find and record overlapping FOVs")

            dfl = []
            for i, ((pos, pu), dftemplate_pos) in enumerate(
                dftemplate.groupby(["Position", "PositionUnique"])
            ):
                xyz = dftemplate_pos[["X", "Y", "Z"]].to_numpy()[0]
                imgsize_um = dftemplate_pos["imgsize_um"].to_numpy()[0]
                template_rectangle = create_rectangle(xyz, imgsize_um)

                for k, ((pos2, pu2), dfmove_pos) in enumerate(
                    dfmove.groupby(["Position", "PositionUnique"])
                ):
                    xyz2 = dfmove_pos[["X", "Y", "Z"]].to_numpy()[0]
                    imgsize_um2 = dfmove_pos["imgsize_um"].to_numpy()[0]
                    move_rectangle = create_rectangle(xyz2, imgsize_um2)

                    overlap = intersection_area(template_rectangle, move_rectangle)
                    if overlap > 0.2:  # require more than 20% overlap
                        feats = {}
                        #             print(pos,pos2,overlap)
                        feats["template_position"] = pos
                        feats["template_position_unique"] = pu
                        feats["template_key"] = keylist[0]
                        feats[
                            "move_position"
                        ] = pos2  # move_position_that_matches_template_position
                        feats["move_position_unique"] = pu2
                        feats["move_key"] = keylist[ki]
                        feats["template_XYZ"] = xyz
                        feats["match_XYZ"] = xyz2
                        feats["xyz_offset_relative_to_template_position"] = (
                            xyz2[0:-1] - xyz[0:-1]
                        )  # move coordinates relative to template coordinates
                        feats["overlap"] = overlap
                        dfl.append(
                            pd.DataFrame(data=feats.values(), index=feats.keys()).T
                        )
            dfoverlap = pd.concat(dfl)
            dfoverlap
            print("dfoverlap.shape", dfoverlap.shape)
            # if no overlap is present for a template position, fill the template_position name as NOMATCH
            # and flag that template_position for that round as having no match with reference
            dfoverlap.set_index(["move_position_unique"], inplace=True)
            dfoverlap["flag-lacks_a_match_with_reference"] = [False] * dfoverlap.shape[
                0
            ]

            # iterate through all the positions in dfmove and ask if that position has a match to dftemplate
            for i, ((move_position_unique, move_position), dfmove_pos) in enumerate(
                dfmove.groupby(["PositionUnique", "Position"])
            ):
                if (
                    move_position_unique not in dfoverlap.index.values
                ):  # if the move position was not identified as having a match
                    # (i.e. move_position_unique not in index of dfoverlap)

                    print("move_position_unique=", move_position_unique)
                    dfoverlap.loc[
                        move_position_unique, "template_position_unique"
                    ] = "NOMATCH"
                    dfoverlap.loc[move_position_unique, "template_position"] = "NOMATCH"
                    dfoverlap.loc[move_position_unique, "move_position"] = move_position
                    dfoverlap.loc[
                        move_position_unique, "flag-lacks_a_match_with_reference"
                    ] = True
                    print(dfoverlap.loc[move_position_unique, :])
            dfoverlap.reset_index(inplace=True)

            # print(dfoverlap)

            # merge the overlapping position info with the "move" dataframe
            # (remember that the first round of matching is the template with itself)
            # dfsub = dfkeep.loc[pd.IndexSlice[[keylist[ki]], :]]
            dfsub = dfmeta.loc[pd.IndexSlice[[keylist[ki]], :]]
            dfm_move = pd.merge(
                dfsub.reset_index(),
                # dfoverlap[["template_position_unique", "move_position_unique", "template_position",
                # "move_position","flag-lacks_a_match_with_reference"]],
                dfoverlap,
                left_on=["PositionUnique"],
                right_on=["move_position_unique"],
                suffixes=("_old", ""),
                how="left",
            )
            print(
                "dfsub.shape, dfoverlap.shape, dfm_move.shape",
                (dfsub.shape, dfoverlap.shape, dfm_move.shape),
            )

            dflall.append(dfm_move)

        dfout = pd.concat(dflall)
        dfout

        # TODO: figure out smart way to keep positions that were imaged too many times
        # plan is to keep the first image

        # dfcount = dfout.reset_index().groupby("template_position").agg("count")
        # number_of_rounds = len(dfout.key.unique())
        # overlapping_poslist = np.unique(
        #     dfcount[dfcount["Position"] >= number_of_rounds].index
        # )

        dfmeta_out = pd.merge(
            dfout,
            dfconfig.loc[[barcode], ["scope", "output_path", "path"]].reset_index(),
            left_on=["barcode", "key", "original_file"],
            right_on=["barcode", "round", "path"],
            how="left",
        )
        print(dfout.shape, dfconfig.shape, dfmeta_out.shape, dfmetaog.shape)

        # important columns
        # index columns are important :
        # ['key', 'template_position'])
        # ['template_position', #position name in template file--this is identical to index column"template_position"
        #  'move_position', #position name in "moving" file
        #  'Position', #position name, this was used for merging
        #  'Scene',
        #  'file',
        #  'parent_file',
        #  'imgsize_pixels',
        #  'ImagePixelDistances',
        #  'totalmagnification',
        #  'channel_dict',
        #  'pixelSizes',
        #  'imgsize_um',
        #  'PlateAnchorPoint',
        #  'PlateReferencePoint',
        #  'X',
        #  'Y',
        #  'Z',
        #  'IsUsedForAcquisition',
        #  'Position_num',
        #  'Well_id',
        #  'X_original',
        #  'X_adjusted',
        #  'Y_original',
        #  'Y_adjusted',
        #  'align_channel',
        #  'barcode',
        #  ]

        # now split scenes and write out all the czi files as ome.tiffs
        output_dir = dfmeta_out["output_path"][0]
        pickle_dir = output_dir + os.sep + "pickles"
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        pickle_name = barcode + "matched_positions_pickle.pickle"
        pickle_path = pickle_dir + os.sep + pickle_name
        print("\n\n" + pickle_path + "\n\n")
        dfmeta_out.to_pickle(os.path.abspath(pickle_path))

        out_csv_path = pickle_path.replace("_pickle", "_csv").replace(".pickle", ".csv")
        dfmeta_out.to_csv(os.path.abspath(out_csv_path))

        # find all positions with extra matches
#         dfg = dfmeta_out.groupby(['template_position_unique','key']).agg('count')
#         ivs = dfg[dfg['Position']>1].index.values
#         dfm = dfmeta_out.set_index(['template_position_unique','key'])
#         flagcols = [x for x in dfm.columns.tolist() if 'flag' in x]
#         print(dfm.loc[ivs,['move_position','Scene','overlapping_positions_within_same_round']+flagcols])


#         # find all positions with no matches
#         print(dfmeta_out.set_index(['template_position_unique','key','PositionUnique']).loc[['NOMATCH']])
