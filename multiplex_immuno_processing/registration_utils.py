import numpy as np
import argparse
import os

from aicsimageio import AICSImage
import matplotlib as plt
import numpy as np
import pandas as pd
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

    # use y-max projections (zx images) for z alignment
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
