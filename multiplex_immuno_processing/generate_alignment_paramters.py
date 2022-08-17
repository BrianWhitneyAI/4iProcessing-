import argparse
import os
import re
import sys

from aicsimageio import AICSImage
import matplotlib as plt
import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader

overwrite = True

# this code computes the alignment parameters to align each scene across the multiple rounds of imaging
# should take a barcode as an argument,
# then this code reads that dataframe pickle for that given barcode
# then it uses the dataframe to determine the reference channel to be used for each of the rounds for alignment
# then it loads all the reference channel images for each round
# then it runs the an alignment algorithm to align all of the rounds
# all positions should be aligned to round 1 (that will be the reference round)
# this will enable all the processing to be run in stages later on as new data is acquire

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
# print(args_dict["barcode"])
# print()




def find_xyz_offset_relative_to_ref(img_list, refimg, ploton=False, verbose=False):
    offset_list = []
    for i in range(len(img_list)):
        test_img = img_list[i]
        (_, _, meanoffset, _,) = find_xyz_offset(
            refimg.copy(), test_img.copy(), ploton=ploton, verbose=verbose
        )
        offset_list.append(meanoffset)

    return offset_list


def find_xyz_offset(target_img, test_img, ploton=False, verbose=False):
    import skimage.exposure as skex

    test_img_rs = test_img.copy()
    target_img_8 = skex.rescale_intensity(
        target_img.copy(), in_range="image", out_range="uint8"
    ).astype("uint8")
    test_img_8 = skex.rescale_intensity(
        test_img_rs.copy(), in_range="image", out_range="uint8"
    ).astype("uint8")
    target_slice = target_img_8.copy().max(axis=0)
    test_slice = test_img_8.copy().max(axis=0)

    if ploton:
        fig, axr = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax = axr.reshape(
            -1,
        )
        plt.sca(ax[0])
        plt.imshow(target_slice, cmap="gray")
        plt.sca(ax[1])
        plt.imshow(test_slice, cmap="gray")
        plt.show()

    # compute yx alignment
    yx_offsetlist, yx_rawmeanoffset = compute_slice_alignment(
        target_slice, test_slice, ploton=False, verbose=verbose
    )

    # specify that there is 0 offset in Z currently because that has not been identified.
    meanoff_yx = list([0]) + list(yx_rawmeanoffset)
    if verbose:
        print("meanoff_in", meanoff_yx)

    # now align the images based on the offset determined above
    in_for_match = [
        target_img_8.copy(),
        test_img_8.copy(),
    ]  # list of images to be matched
    offset_list1 = [list(np.round(meanoff_yx))]  # list of offsets

    # now match the images and return the matches
    match_listxy = return_aligned_img_list_new(
        in_for_match, offset_list1, verbose=verbose
    )

    # returned matches
    target_img_matched_8 = match_listxy[0]
    test_img_matched_8 = match_listxy[1]

    # compute xz projections to be used for z alignment
    target_slice = target_img_8.copy().max(axis=1)
    test_slice = test_img_8.copy().max(axis=1)

    # get y-max projections (zx images) for z alignment
    zx_offsetlist, zx_rawmeanoffset = compute_slice_alignment(
        target_slice,
        test_slice,
        ax1factor=1.5,
        ax2factor=3,
        nxsteps=5,
        nysteps=5,
        ploton=False,
        verbose=False,
    )

    meanoffset = [zx_rawmeanoffset[0]] + list(yx_rawmeanoffset)

    in_for_match_zx = [target_img_8.copy(), test_img_8.copy()]
    offset_list_zx = [list(np.round(meanoffset))]

    if verbose:
        print("meanoffset", meanoffset)

    match_list_zx = return_aligned_img_list_new(
        in_for_match_zx, offset_list_zx, verbose=verbose
    )

    target_img_matched_8 = match_list_zx[0]
    test_img_matched_8 = match_list_zx[1]

    if verbose:
        print(target_img_matched_8.shape, test_img_matched_8.shape)

    # if ploton:
    #     plot_overlays(target_img_matched_8, test_img_matched_8)

    cropoffset = []
    return target_img_matched_8, test_img_matched_8, meanoffset, cropoffset


def compute_slice_alignment(
    target_slice,
    test_slice,
    nxsteps=10,
    nysteps=10,
    ax1factor=2,
    ax2factor=2,
    ploton=False,
    verbose=False,
):
    import cv2 as cv

    #     ww = int(np.ceil(target_slice.shape[0]/ax1factor)) # width of matching crop
    #     hh = int(np.ceil(target_slice.shape[1]/ax2factor)) # height of matching crop
    #     xg = int(np.ceil(target_slice.shape[0])/100) # gap from left and right edge
    #     yg = int(np.ceil(target_slice.shape[1])/100) #gap from top and bottom
    #     xsteplist = np.int32(np.linspace(xg,target_slice.shape[0] - ww - xg, nxsteps))
    #     ysteplist = np.int32(np.linspace(yg,target_slice.shape[1] - hh - yg, nysteps))

    ww = int(np.ceil(test_slice.shape[0] / ax1factor))  # width of matching crop
    hh = int(np.ceil(test_slice.shape[1] / ax2factor))  # height of matching crop
    xg = int(np.ceil(test_slice.shape[0]) / 100)  # gap from left and right edge
    yg = int(np.ceil(test_slice.shape[1]) / 100)  # gap from top and bottom

    xsteplist = np.int32(np.linspace(xg, test_slice.shape[0] - ww - xg, nxsteps))
    ysteplist = np.int32(np.linspace(yg, test_slice.shape[1] - hh - yg, nysteps))

    while target_slice.shape[0] < (ww * 1.2):
        ax1factor = ax1factor * 1.2
        ww = int(np.ceil(test_slice.shape[0] / ax1factor))  # width of matching crop

    while target_slice.shape[1] < (hh * 1.2):
        ax2factor = ax2factor * 1.2
        hh = int(np.ceil(test_slice.shape[1] / ax2factor))  # height of matching crop

    if verbose:
        print("ww,hh,xg,yg", ww, hh, xg, yg)
        print("xsteplist,ysteplist", xsteplist, ysteplist)
        print(
            "target_slice.shape,test_slice.shape", target_slice.shape, test_slice.shape
        )

    offsetlist = []
    reslist = []
    for xi in xsteplist:
        for yi in ysteplist:
            xx = [xi, xi + ww]
            yy = [yi, yi + hh]

            #             img_gray = test_slice
            #             template = target_slice[xx[0]:xx[1],yy[0]:yy[1]]

            img_gray = target_slice
            template = test_slice[xx[0] : xx[1], yy[0] : yy[1]]

            w, h = template.shape[::-1]
            res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)

            loc = np.where(res >= np.max(res))

            offset = [loc[0] - xx[0], loc[1] - yy[0]]
            offset = [
                x * -1 for x in offset
            ]  # this is necessary to ensure it is correct direction
            offsetlist.append(offset)
            reslist.append(np.max(res))
            if ploton:
                test_slice = target_slice.copy()
                new_target = cv.cvtColor(test_slice, cv.COLOR_BGR2RGB)

                for pt in zip(*loc[::-1]):
                    cv.rectangle(
                        new_target, pt, (pt[0] + w, pt[1] + h), (255, 255, 0), 5
                    )

                plt.figure(figsize=(2, 2))
                plt.imshow(template)
                plt.show()
                plt.figure(figsize=(4, 3))
                plt.imshow(new_target)
                plt.show()

    offsetlist = [
        x for x, y in zip(offsetlist, reslist) if y >= np.nanpercentile(reslist, 95)
    ]

    if verbose is True:
        print("res", res)

    # now compute the mean offset
    rawmeanoffset = [
        np.nanmean([x[0] for x in offsetlist]),
        np.nanmean([x[1] for x in offsetlist]),
    ]

    return offsetlist, rawmeanoffset


def return_aligned_img_list_new(
    img_list, offset_list, subpixel=False, verbose=False, return_roi=False
):
    offset_array = np.asarray(offset_list)
    offset_cum_sum = np.cumsum(
        offset_array, axis=0
    )  # this gives you the distance from img1

    # determine the max depth width and height required for the matched images
    # this is done by creating overlaps of the img rois after moved by their offsets relative to the first image
    # all_offs is all offsets relative to img_list[0]
    all_offs = np.vstack((np.zeros(offset_array[0].shape), offset_cum_sum))

    if subpixel is False:
        all_offs = np.round(all_offs, 0)

    roilist = []
    for i, img in enumerate(img_list):
        off = all_offs[i]
        roi = np.zeros(len(img.shape) * 2)
        for i in range(len(img.shape)):
            roi[(i * 2) : (i + 1) * 2] = np.asarray([0, img.shape[i]]) - off[i]
        roilist.append(roi)

    rois = np.asarray(roilist)
    maxroi = np.max(
        rois, axis=0
    )  # maximize the distance from origin (e.g. inner lower left corner)
    minroi = np.min(
        rois, axis=0
    )  # minimize the distance from origin for the outer portions of roi (e.g. outer upper right corner)
    new_roi = np.zeros(len(img_list[0].shape) * 2)
    new_roi[np.asarray([0, 2, 4])] = maxroi[np.asarray([0, 2, 4])]
    new_roi[np.asarray([1, 3, 5])] = minroi[np.asarray([1, 3, 5])]

    d = new_roi[1] - new_roi[0]
    w = new_roi[3] - new_roi[2]
    h = new_roi[5] - new_roi[4]
    new_dwh = np.asarray([d, w, h])

    # now create rois for the matched regions
    nz, ny, nx = new_roi[0], new_roi[2], new_roi[4]
    roi_adj_list = []
    for roi in roilist:
        z, y, x = roi[0], roi[2], roi[4]
        az, ay, ax = nz - z, ny - y, nx - x  # adjusted
        roi_adj = np.round(np.asarray([az, az + d, ay, ay + w, ax, ax + h])).astype(
            "int32"
        )
        roi_adj_list.append(roi_adj)

    if verbose:
        print("imgshape", img_list[0].shape)
        print("roilist", roilist)
        print("new_dwh", new_dwh)
        print("roi_adj_list", roi_adj_list)

    match_list = []
    roi_list = []
    for iv, img in enumerate(img_list):
        roi = roi_adj_list[iv]
        if verbose:
            print("all_offs", all_offs[iv])

        if verbose:
            print(roi)
        match = img[
            roi[0] : roi[1],
            roi[2] : roi[3],
            roi[4] : roi[5],
        ]
        match_list.append(match)

        roi_list.append(roi)
    if return_roi:
        return roi_list
    else:
        return match_list






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
    for Position in template_position_list:
    # for Position in ['P6', 'P3', 'P12']: #go one position by position, since you need offsets per position
        print("POSITION = ", Position)
        keeplist = []
        
        # need to define the keylist for each position, since some positions may not be imaged every round
        keylist = dfall.set_index('template_position').loc[Position,'key'].unique()

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

        keylist = dfimg.index.values

        img_list = []
        for key in keylist:
            align_channel = dfimg.loc[key, "align_channel"]
            imgstack = dfimg.loc[key, "img"]
            img_list.append(imgstack.copy())
        print(len(img_list))

        reference_round_key = "Round 1"
        refimg = dfimg.loc[reference_round_key, "img"]
        alignment_offsets_xyz_list = find_xyz_offset_relative_to_ref(
            img_list, refimg=refimg, ploton=False, verbose=False
        )
        # match_list = align_helper.return_aligned_img_list_new(img_list,offset_list)
        print("alignment_offsets_xyz_list\n", alignment_offsets_xyz_list)
        # unmatch_list = align_helper.return_aligned_img_list_new(img_list,np.asarray(offset_list)*0)

        dfimg["alignment_offsets_xyz"] = alignment_offsets_xyz_list
        dfimg["template_position"] = [Position] * dfimg.shape[0]
        dfout_p = dfimg[["template_position", "align_channel", "alignment_offsets_xyz"]]
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
        csv_name = pickle_name.replace('_pickle','_csv').replace('.pickle','.csv')
        out_csv_path = pickle_dir + os.sep +  csv_name
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
    csv_name = pickle_name.replace('_pickle','_csv').replace('.pickle','.csv')
    out_csv_path = pickle_dir + os.sep +  csv_name
    dfout.to_csv(os.path.abspath(out_csv_path))
