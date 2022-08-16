import math
import os
from typing import Tuple

import numpy as np
import SimpleITK as sitk
import skimage.exposure as exp
import skimage.transform as tf
from scipy.ndimage import shift
from skimage.feature import ORB, match_descriptors
from skimage.filters import gaussian, median, threshold_otsu
from skimage.measure import ransac
import pandas as pd
import ast
import tifffile
import yaml
from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio import AICSImage, writers


print("imports loaded")

def rescale_image(image: np.ndarray, scale_factor_xy: float, scale_factor_z: float):
    """Upsample/Downsample the image to match voxel dimensions of the other image.
    Parameters
    ------------
        - image: 3D or 4D image to rescale
        - scale_factor_xy: Upsample/downsample rate in x and y
        - scale_factor_z: Upsample/downsample rate in z
    Returns
    ------------
        - image: rescaled image
    """

    if image.ndim == 3:
        return tf.resize(
            image,
            (
                int(round(image.shape[0] * scale_factor_z)),
                int(round(image.shape[1] * scale_factor_xy)),
                int(round(image.shape[2] * scale_factor_xy)),
            ),
            preserve_range=True,
        ).astype(np.uint16)
    if image.ndim == 4:
        return tf.resize(
            image,
            (
                image.shape[0],
                int(round(image.shape[1] * scale_factor_z)),
                int(round(image.shape[2] * scale_factor_xy)),
                int(round(image.shape[3] * scale_factor_xy)),
            ),
            preserve_range=True,
        ).astype(np.uint16)


def perform_alignment(
    source: np.ndarray,
    target: np.ndarray,
    smaller_fov_modality: str,
    scale_factor_xy: float,
    scale_factor_z: float,
    source_alignment_channel: int,
    target_alignment_channel: int,
    source_output_channel: list,
    target_output_channel: list,
    prealign_z: bool,
    denoise_z: bool,
    use_refinement: bool=False,
    save_composite: bool=False,
):
    # function modififed from https://github.com/AllenCell/aics_tf_registration/blob/master/aics_tf_registration/core/alignment.py
    """Wrapper function for all of the steps necessary to calculate alignment.
    Parameters
    ------------
        - source: low-res modality image
        - target: high-res modality image
        - smaller_fov_modality: which modality has the smaller field of view
        - scale_factor_xy: upsample/downsample rate to match image scales in x and y
        - scale_factor_z: upsample/downsample rate to match image scales in z
        - source_alignment_channel: source image channel used for calculating alignment
        - target_alignment_channel: target image channel used for calculating alignment
        - source_output_channel: source image channel to apply alignment on
        - target_output_channel: target image channel to apply alignment on
        - prealign: whether to calculate intitial estimate of z-alignment
        - denoise_z: denoise z-stacks prior to z-alignment
        - use_refinement: refine alignment by repeating in the target image resolution
        - save_composite: save composite image of final alignment
    Returns
    -----------
        - source_aligned: aligned source image
        - target_aligned: aligned target image
    """

    # split 4d images into alignment and output images if necessary
    if source.ndim == 4:
        source_align = source[source_alignment_channel]
        source_out = source[source_output_channel]
    else:
        source_align = source_out = source
    if target.ndim == 4:
        target_align = target[target_alignment_channel]
        target_out = target[target_output_channel]
    else:
        target_align = target_out = target

    print("Assigning source and target images")
    # Assign source and target images to fixed and moving images
    # In my case the source image will always be round 1 and the target image will have to be a different round


    if smaller_fov_modality == "source":
        print("source -> moving      target -> fixed")
        moving = exp.rescale_intensity(source_align, out_range=np.uint16).astype(
            np.uint16
        )
        fixed = exp.rescale_intensity(target_align, out_range=np.uint16).astype(
            np.uint16
        )
        # moving = source_align
        # fixed = target_align
        moving_out = source_out
        fixed_out = target_out
    else:
        print("target -> moving      source -> fixed")
        moving = exp.rescale_intensity(target_align, out_range=np.uint16).astype(
            np.uint16
        )
        fixed = exp.rescale_intensity(source_align, out_range=np.uint16).astype(
            np.uint16
        )
        # moving = target_align
        # fixed = source_align
        moving_out = target_out
        fixed_out = source_out

    # rescale moving image to match fixed image dimensions
    print(f"Fixed image has a shape of {fixed.shape}")
    print(f"Moving image began with a shape of {moving.shape}")
    moving_scaled = rescale_image(moving, scale_factor_xy, scale_factor_z)
    print(f"Moving image has been rescaled to shape {moving_scaled.shape}")

    # pad or crop z layers from moving image to match fixed image
    init_z_padding = fixed.shape[0] - moving_scaled.shape[0]
    print(f"Initial z-padding to moving image: {init_z_padding}")
    if init_z_padding > 0:
        print("Padding moving image")
        moving_scaled_adjust_z = np.pad(
            moving_scaled,
            (
                (int(init_z_padding // 2), int(math.ceil(init_z_padding / 2))),
                (0, 0),
                (0, 0),
            ),
            mode="constant",
        )
    elif init_z_padding < 0:
        print("Clipping moving image")
        clip_start = int(abs(init_z_padding) // 2)
        clip_end = int(moving_scaled.shape[0] - int(math.ceil(init_z_padding / 2)))
        moving_scaled_adjust_z = moving_scaled[clip_start:clip_end, :, :]
    elif init_z_padding == 0:
        moving_scaled_adjust_z = moving_scaled.copy()
    msg = f"Final rescaled shape of moving image is {moving_scaled_adjust_z.shape}"
    print(msg)

    # perform initial 2d alignment
    tifffile.imwrite("fixed.tiff", fixed)
    tifffile.imwrite("moving_scaled_adjust_z.tiff", moving_scaled_adjust_z)

    print("Beginning calculation of rigid offset in x and y")
    fixed_2dAlign_offset_x, fixed_2dAlign_offset_y = align_xy(
        fixed, moving_scaled_adjust_z
    )
    if fixed_2dAlign_offset_x is not None:
        print("2d alignment successful")
        print(f"x offset: {fixed_2dAlign_offset_x - 5}")
        print(f"y offset: {fixed_2dAlign_offset_y - 5}")
    else:
        return None, None, None

    buffered_offset_y = int(fixed_2dAlign_offset_y - 5)
    buffered_offset_x = int(fixed_2dAlign_offset_x - 5)

    if buffered_offset_x < 0 or buffered_offset_y < 0:
        msg1 = "A offset from 2d-alignment is negative. "
        msg2 = "Moving image is not wholly in fixed FOV. Consider using pre-cropping"
        print(msg1 + msg2)
        return None, None, None

    # prepare images for alignment with itk
    moving_scaled_adjust_for_itk = np.zeros(
        (
            moving_scaled_adjust_z.shape[0],
            moving_scaled_adjust_z.shape[1] + 10,
            moving_scaled_adjust_z.shape[2] + 10,
        )
    )
    moving_scaled_adjust_for_itk[:, 5:-5, 5:-5] = moving_scaled_adjust_z[:, :, :]

    fixed_pre_crop_for_itk = fixed[
        :,
        buffered_offset_y : buffered_offset_y + moving_scaled_adjust_for_itk.shape[1],
        buffered_offset_x : buffered_offset_x + moving_scaled_adjust_for_itk.shape[2],
    ]

    # perform z alignment with itk
    print("Beginning calculation of rigid offset in z")
    fixed_addition_offset_z, fixed_addition_offset_x, fixed_addition_offset_y = align_z(
        fixed_pre_crop_for_itk,
        moving_scaled_adjust_for_itk,
        prealign_z,
        denoise_z,
    )
    if fixed_addition_offset_z is not None:
        print("z-alignment sucessful")
        print(f"z offset: {fixed_addition_offset_z}")
        print(f"additional x offset: {fixed_addition_offset_x}")
        print(f"additional y offset: {fixed_addition_offset_y}")
    else:
        print("z-alignment failed")
        return None, None, None

    # refine alignment and apply to output channel
    print("Beginning finalization of alignment")
    if use_refinement:
        print("Refinement enabled. This part might take a while ...")

    fixed_final = moving_final = None
    moving_crops = fixed_crops = None

    print("before finalize alignment function")
    print(f"fixed img is of shape {np.shape(fixed)}")
    print(f"moving img is of shape {np.shape(moving)}")
    print(f"moving scaled shape is {moving_scaled.shape}")
    print(f"moving_scaled_adjust_z shape is {moving_scaled_adjust_z.shape}")
    print(f"inti z padding is {init_z_padding}")
    print(f"offset z is {fixed_addition_offset_z}")
    print(f"offset y is {fixed_addition_offset_y}")
    print(f"offset x is {fixed_addition_offset_x}")
    print(f"align offset in y is {fixed_2dAlign_offset_y}")
    print(f"align offset in x is {fixed_2dAlign_offset_x}")

    fixed, moving, final_z_offset, final_y_offset, final_x_offset = finalize_alignment(
        fixed,
        moving,
        moving_scaled.shape,
        moving_scaled_adjust_z.shape,
        init_z_padding,
        fixed_addition_offset_z,
        fixed_addition_offset_y,
        fixed_addition_offset_x,
        fixed_2dAlign_offset_y,
        fixed_2dAlign_offset_x,
        scale_factor_xy,
        scale_factor_z,
    )
    print("after final filignment function")
    print(f"fixed img is of shape {np.shape(fixed)}")
    print(f"moving img is of shape {np.shape(moving)}")

    if use_refinement:
        if fixed.size > moving.size:
            moving_crops, fixed_crops = final_refinement(
                moving, fixed, scale_factor_xy, scale_factor_z, 15, 3
            )
        else:
            fixed_crops, moving_crops = final_refinement(
                fixed, moving, 1 / scale_factor_xy, 1 / scale_factor_z, 15, 3
            )

        if fixed_crops is None:
            return None, None, None

    if not moving_out.ndim == 4:
        moving_out = np.expand_dims(moving_out, axis=0)
    if not fixed_out.ndim == 4:
        fixed_out = np.expand_dims(fixed_out, axis=0)

    for channel in range(moving_out.shape[0]):
        moving_chan = moving_out[channel]
        fixed_chan = fixed_out[channel]

        fixed_temp, moving_temp, moving_z_offset, moving_y_offset, moving_x_offset = finalize_alignment(
            fixed_chan,
            moving_chan,
            moving_scaled.shape,
            moving_scaled_adjust_z.shape,
            init_z_padding,
            fixed_addition_offset_z,
            fixed_addition_offset_y,
            fixed_addition_offset_x,
            fixed_2dAlign_offset_y,
            fixed_2dAlign_offset_x,
            scale_factor_xy,
            scale_factor_z,
        )

        if not (fixed_temp is None) and not (moving_crops is None):
            moving_temp = moving_temp[
                moving_crops[0][0] : moving_crops[0][1],
                moving_crops[1][0] : moving_crops[1][1],
                moving_crops[2][0] : moving_crops[2][1],
            ]
            fixed_temp = fixed_temp[
                fixed_crops[0][0] : fixed_crops[0][1],
                fixed_crops[1][0] : fixed_crops[1][1],
                fixed_crops[2][0] : fixed_crops[2][1],
            ]

        if (fixed_final is None) and not (fixed_temp is None):
            fixed_final = np.expand_dims(fixed_temp, axis=0)
            moving_final = np.expand_dims(moving_temp, axis=0)
        elif not (fixed_temp is None):
            fixed_final = np.concatenate(
                (fixed_final, np.expand_dims(fixed_temp, axis=0)), axis=0
            )
            moving_final = np.concatenate(
                (moving_final, np.expand_dims(moving_temp, axis=0)), axis=0
            )
    if fixed_final is None:
        print("Finalization failed!")
        return None, None, None

    print("Finalization complete!")

    if smaller_fov_modality == "source":
        source_aligned = moving_final
        target_aligned = fixed_final
    else:
        source_aligned = fixed_final
        target_aligned = moving_final

    if save_composite:
        print("Creating alignment composite image")
        # import pdb; pdb.set_trace()
        if use_refinement:
            moving = moving[
                moving_crops[0][0] : moving_crops[0][1],
                moving_crops[1][0] : moving_crops[1][1],
                moving_crops[2][0] : moving_crops[2][1],
            ]
            fixed = fixed[
                fixed_crops[0][0] : fixed_crops[0][1],
                fixed_crops[1][0] : fixed_crops[1][1],
                fixed_crops[2][0] : fixed_crops[2][1],
            ]
        if smaller_fov_modality == "source":
            source_comp = moving[..., np.newaxis]
            target_comp = fixed[..., np.newaxis]
        else:
            source_comp = fixed[..., np.newaxis]
            target_comp = moving[..., np.newaxis]

        try:
            composite = np.concatenate(
                (
                    tf.resize(
                        source_comp,
                        target_comp.shape,
                        preserve_range=True,
                        anti_aliasing=True,
                    ),
                    target_comp,
                    np.zeros_like(target_comp),
                ),
                axis=3,
            )
            print("Composite image created")
        except (RuntimeError, TypeError, ValueError):
            composite = None
            print("Error in creating composite image")
    else:
        composite = None
        print("No composite image created")

    return source_aligned, target_aligned, composite, final_z_offset, final_y_offset, final_x_offset

def align_xy(fixed: np.ndarray, moving: np.ndarray):
    """Perform alignment of the images in 2d.
    Parameters
    ------------
        - fixed: image with larger field of view
        - moving: image with smaller field of view
    Return
    ------------
        - fixed_2dAlign_offset_x: rigid offset in x
        - fixed_2dAlign_offset_y: rigid offset in y
    """


    fixed_proj = np.max(fixed, axis=0)
    moving_proj = np.max(moving, axis=0)

    # Intensity rescaling and contrast enhancement
    inf, sup = np.percentile(fixed_proj, [5, 95])
    fixed_proj = np.clip(fixed_proj, inf, sup)
    fixed_proj = gaussian(fixed_proj)
    fixed_proj = median(fixed_proj)
    fixed_proj = exp.rescale_intensity(fixed_proj, out_range=(0, 65535)).astype(
        np.uint16
    )

    inf, sup = np.percentile(moving_proj, [5, 95])
    moving_proj = np.clip(moving_proj, inf, sup)
    moving_proj = gaussian(moving_proj)
    moving_proj = median(moving_proj)
    moving_proj = exp.rescale_intensity(moving_proj, out_range=(0, 65535)).astype(
        np.uint16
    )

    # Extract keypoints and descriptors for each image
    try:
        descriptor_extractor = ORB(n_keypoints=2500)
    except (RuntimeError, TypeError, ValueError):
        return None, None

    # fixed image
    try:
        descriptor_extractor.detect_and_extract(fixed_proj)
        keypoints_fix = descriptor_extractor.keypoints
        descriptors_fix = descriptor_extractor.descriptors
    except (RuntimeError, TypeError, ValueError):
        return None, None

    # moving image
    try:
        descriptor_extractor.detect_and_extract(moving_proj)
        keypoints_mov = descriptor_extractor.keypoints
        descriptors_mov = descriptor_extractor.descriptors
    except (RuntimeError, TypeError, ValueError):
        return None, None

    # match descriptors/keypoints in images
    trial_fail = True
    for trial_idx in range(10):
        matches = match_descriptors(
            descriptors_fix,
            descriptors_mov,
            metric="euclidean",
            max_ratio=0.95,
            cross_check=True,
        )

        # Check if sufficient number of keypoints have been matched
        if len(matches[:, 0]) < 5:
            continue

        # Estimate initial Similarity Transform for moving image
        src = keypoints_mov[matches[:, 1]][:, ::-1]
        dst = keypoints_fix[matches[:, 0]][:, ::-1]
        model_robust, _ = ransac(
            (src, dst),
            tf.EuclideanTransform,
            min_samples=3,
            residual_threshold=2,
            max_trials=200,
        )


        if (
            abs(model_robust.params[0, 0] - 1) > 0.1
            or abs(model_robust.params[1, 1] - 1) > 0.1
            or abs(model_robust.params[0, 1]) > 0.1
            or abs(model_robust.params[1, 0]) > 0.1
        ):
            # if the fitted model has a lot rotation, then try again

            continue

        if math.isnan(model_robust.params[0, 2]) or math.isnan(
            model_robust.params[1, 2]
        ):
            # fail

            continue

        fixed_2dAlign_offset_x = round(model_robust.params[0, 2])
        fixed_2dAlign_offset_y = round(model_robust.params[1, 2])

        trial_fail = False

        break

    if trial_fail:
        return None, None

    return fixed_2dAlign_offset_x, fixed_2dAlign_offset_y

def align_z(fixed: np.ndarray, moving: np.ndarray, prealign: bool, denoise: bool):
    """Align the images in the z-plane.
    Parameters
    ------------
        - fixed: image with larger field of view
        - moving: image with smaller field of view
        - prealign: whether to initially estimate alignment by overlapping segmentations
        - denoise: "denoise" image through 10th & 90th percentile clipping
    Return
    ------------
        - fixed_addition_offset_z: rigid offset in z
        - fixed_addition_offset_x: rigid offset in x
        - fixed_addition_offset_y: rigid offset in y
    """

    if denoise:
        inf, sup = np.percentile(fixed, [5, 95])

        fixed = np.clip(fixed, inf, sup)
        fixed = exp.rescale_intensity(fixed, out_range=np.uint8)

        inf, sup = np.percentile(moving[:, 10:-10, 10:-10], [5, 95])

        moving = np.clip(moving, inf, sup)
        moving = exp.rescale_intensity(moving, out_range=np.uint8)

    if prealign:

        moving_threshold = threshold_otsu(moving[:, 10:-10, 10:-10])
        fixed_threshold = threshold_otsu(fixed)

        moving_otsu = moving >= moving_threshold
        fixed_otsu = fixed >= fixed_threshold

        moving_z_above_threshold = np.argwhere(
            np.max(np.max(moving_otsu, axis=1), axis=1) > 0
        ).squeeze()
        fixed_z_above_threshold = np.argwhere(
            np.max(np.max(fixed_otsu, axis=1), axis=1) > 0
        ).squeeze()

        moving_estimated_center_z = (
            np.max(moving_z_above_threshold) + np.min(moving_z_above_threshold)
        ) // 2
        fixed_estimated_center_z = (
            np.max(fixed_z_above_threshold) + np.min(fixed_z_above_threshold)
        ) // 2
        estimated_z_offset = int(fixed_estimated_center_z - moving_estimated_center_z)

        moving = shift(moving, (estimated_z_offset, 0, 0))

    else:
        estimated_z_offset = 0

    # move to itk
    fixed_itk = sitk.GetImageFromArray(
        exp.rescale_intensity(fixed.astype(np.float32), out_range=(0, 255)).astype(
            np.uint8
        )
    )
    fixed_itk = sitk.Cast(fixed_itk, sitk.sitkFloat32)
    moving_itk = sitk.GetImageFromArray(
        exp.rescale_intensity(moving.astype(np.float32), out_range=(0, 255)).astype(
            np.uint8
        )
    )
    moving_itk = sitk.Cast(moving_itk, sitk.sitkFloat32)

    # Initialize ITK-based image registration parameters
    R = sitk.ImageRegistrationMethod()
    R.SetOptimizerAsRegularStepGradientDescent(5, 0.01, 50)
    R.SetInitialTransform(sitk.TranslationTransform(fixed_itk.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.25)

    # Estimate z-alignment transform
    try:
        outTx = R.Execute(fixed_itk, moving_itk)
    except RuntimeError:
        return None, None, None

    if "SITK_NOSHOW" not in os.environ:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_itk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(100)
        resampler.SetTransform(outTx)
        resampler.Execute(moving_itk)
    else:
        return None, None, None

    shift_vector = outTx.GetParameters()

    fixed_addition_offset_z = round(shift_vector[2]) - estimated_z_offset + 1
    fixed_addition_offset_y = round(shift_vector[1])
    fixed_addition_offset_x = round(shift_vector[0])


    return fixed_addition_offset_z, fixed_addition_offset_x, fixed_addition_offset_y

def finalize_alignment(
    fixed: np.ndarray,
    moving_orig: np.ndarray,
    moving_scaled_shape: Tuple[int],
    moving_adjust_z_shape: Tuple[int],
    init_z_padding: int,
    fixed_addition_offset_z: int,
    fixed_addition_offset_y: int,
    fixed_addition_offset_x: int,
    fixed_2dAlign_offset_y: int,
    fixed_2dAlign_offset_x: int,
    scale_factor_xy: float,
    scale_factor_z: float,
):
    """Use refineuse_refinement-force search adjust the final 3d alignment.
    Parameters
    ------------
        - fixed: image with larger field of view
        - moving_orig: image with smaller field of view
        - moving_scaled_shape: image dimensions of rescaled smaller FOV
        - moving_adjust_z_shape: image dimensions after adjustments to stack size
        - init_z_padding: number of stacks padded or clipped from rescaled image
        - fixed_addition_offset_z: rigid offset in z from itk alignment
        - fixed_addition_offset_y: rigid offset in y from itk alignment
        - fixed_addition_offset_x: rigid offest in x from itk alignment
        - fixed_2dAlign_offset_y: rigid offset in y from 2d alignment
        - fixed_2dAlign_offset_x: rigid offest in x from 2d alignment
        - scale_factor_xy: upsample/downsample rate for x and y
        - scale_factor_z: upsample/downsample rate for z
    Returns
    ------------
        - fixed_final: aligned and cropped fixed image
        - moving_final: aligned and cropped moving image
    """

    if init_z_padding > 0:
        fixed_z_offset = int(round(init_z_padding // 2)) - fixed_addition_offset_z
        fixed_y_offset = int(round(fixed_2dAlign_offset_y)) - fixed_addition_offset_y
        fixed_x_offset = int(round(fixed_2dAlign_offset_x)) - fixed_addition_offset_x



        moving_z_bot = 0
        moving_zz_top = moving_orig.shape[0] + 1

        if fixed_z_offset < 0:
            moving_z_bot += round(abs(fixed_z_offset) / scale_factor_z)

        if fixed_z_offset + moving_scaled_shape[0] > fixed.shape[0]:
            moving_zz_top -= int(
                round(
                    abs(fixed.shape[0] - (fixed_z_offset + moving_scaled_shape[0]))
                    / scale_factor_z
                )
            )


        fixed_final = fixed[
            np.max([fixed_z_offset, 0]) : np.min(
                [fixed_z_offset + moving_scaled_shape[0], fixed.shape[0]]
            ),
            fixed_y_offset : fixed_y_offset + moving_scaled_shape[1],
            fixed_x_offset : fixed_x_offset + moving_scaled_shape[2],
        ]
        moving_final = moving_orig[moving_z_bot:moving_zz_top, :, :]
    else:

        fixed_z_offset = -fixed_addition_offset_z
        fixed_y_offset = int(round(fixed_2dAlign_offset_y)) - fixed_addition_offset_y
        fixed_x_offset = int(round(fixed_2dAlign_offset_x)) - fixed_addition_offset_x

        print(f"fixed z offset new is {fixed_z_offset}")
        print(f"fixed y offset new is {fixed_y_offset}")
        print(f"fixed x offset new is {fixed_x_offset}")

        moving_z_bot = int(
            round(math.floor(abs(init_z_padding) / (2 * scale_factor_z)))
        )
        moving_zz_top = (
            moving_orig.shape[0]
            - int(round(math.ceil(abs(init_z_padding) / (2 * scale_factor_z))))
            - 2
        )

        if fixed_z_offset < 0:
            moving_z_bot += int(round(abs(fixed_z_offset) / scale_factor_z))

        if fixed_z_offset + moving_adjust_z_shape[0] > fixed.shape[0]:
            moving_zz_top -= int(
                round(
                    abs(fixed.shape[0] - (fixed_z_offset + moving_adjust_z_shape[0]))
                    / scale_factor_z
                )
            )


        fixed_final = fixed[
            np.max([fixed_z_offset, 0]) : np.min(
                [fixed_z_offset + moving_scaled_shape[0], fixed.shape[0]]
            ),
            fixed_y_offset : fixed_y_offset + moving_scaled_shape[1],
            fixed_x_offset : fixed_x_offset + moving_scaled_shape[2],
        ]
        moving_final = moving_orig[moving_z_bot:moving_zz_top, :, :]

    return fixed_final, moving_final, fixed_z_offset, fixed_y_offset, fixed_x_offset

def final_refinement(
    lr: np.ndarray,
    hr: np.ndarray,
    scale_factor_xy: float,
    scale_factor_z: float,
    min_subcrop_xy: int,
    min_subcrop_z: int,
    error_thresh: float = 0.01,
):
    """Adjust the final 3d alignment by repeating alignment in high resolution image scale.
    Parameters
    ------------
        - lr: low-res (source) image after initial alignment
        - hr: high-res (target) image after initial alignment
        - scale_factor_xy: upsample/downsample rate for x and y
        - scale_factor_z: upsample/downsample rate for z
        - min_subcrop_xy: minimum number of pixels to crop to match scale factor
        - min_subcrop_z: minimum number of pixels to crop to match scale factor
        - error_thresh: maximum error in scale factors tolerated for final image
    Returns
    ------------
        - fixed_final: aligned and cropped fixed image
        - moving_final: aligned and cropped moving image
    """


    n_xy = n_z = 1
    while (
        n_xy * scale_factor_xy < min_subcrop_xy
        or abs((round(n_xy * scale_factor_xy) / n_xy) - scale_factor_xy) > error_thresh
    ):
        n_xy += 1
    while (
        n_z * scale_factor_z < min_subcrop_z
        or abs((round(n_xy * scale_factor_xy) / n_xy) - scale_factor_xy) > error_thresh
    ):
        n_z += 1

    subcrop_xy = int(round(n_xy * scale_factor_xy))
    subcrop_z = int(round(n_z * scale_factor_z))

    print(f"lr starts with shape {lr.shape}")
    print(f"hr starts with shape {hr.shape}")

    print(f"z scale factor: {scale_factor_z}")
    print(f"xy scale factor: {scale_factor_xy}")
    print(f"Starting crop in z: {n_z} (lr) {subcrop_z} (hr)")
    print(f"Effective z-scaling is: {(subcrop_z/n_z):.3f}")
    print(f"Starting crop in xy: {n_xy} (lr) {subcrop_xy} (hr)")
    print(f"Effective xy-scaling is: {(subcrop_xy/n_xy):.3f}")

    lr_cropped = lr[n_z:-n_z, n_xy:-n_xy, n_xy:-n_xy]
    try:
        lr_cropped_rescaled = rescale_image(lr_cropped, scale_factor_xy, scale_factor_z)
    except RuntimeError:
        return None, None

    lr_cropped_rescaled_padded = np.pad(
        lr_cropped_rescaled,
        pad_width=(
            (subcrop_z, subcrop_z),
            (subcrop_xy, subcrop_xy),
            (subcrop_xy, subcrop_xy),
        ),
    )

    hr_itk = sitk.GetImageFromArray(
        exp.rescale_intensity(hr.astype(np.float32), out_range=(0, 255)).astype(
            np.uint8
        )
    )
    hr_itk = sitk.Cast(hr_itk, sitk.sitkFloat32)
    lr_itk = sitk.GetImageFromArray(
        exp.rescale_intensity(
            lr_cropped_rescaled_padded.astype(np.float32), out_range=(0, 255)
        ).astype(np.uint8)
    )
    lr_itk = sitk.Cast(lr_itk, sitk.sitkFloat32)

    # Initialize ITK-based image registration parameters
    R = sitk.ImageRegistrationMethod()
    R.SetOptimizerAsRegularStepGradientDescent(5, 0.01, 50)
    R.SetInitialTransform(sitk.TranslationTransform(hr_itk.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.25)

    # Estimate z-alignment transform
    try:
        outTx = R.Execute(hr_itk, lr_itk)
    except RuntimeError:
        return None, None

    if "SITK_NOSHOW" not in os.environ:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(hr_itk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(100)
        resampler.SetTransform(outTx)
        resampler.Execute(lr_itk)
    else:
        return None, None

    shift_vector = outTx.GetParameters()

    hr_offset_z = int(round(shift_vector[2]))
    hr_offset_y = int(round(shift_vector[1]))
    hr_offset_x = int(round(shift_vector[0]))

    hr_z_start = subcrop_z - hr_offset_z
    hr_z_stop = (
        subcrop_z - hr_offset_z + int(round(lr_cropped.shape[0] * scale_factor_z))
    )
    hr_y_start = subcrop_xy - hr_offset_y
    hr_y_stop = (
        subcrop_xy - hr_offset_y + int(round(lr_cropped.shape[1] * scale_factor_xy))
    )
    hr_x_start = subcrop_xy - hr_offset_x
    hr_x_stop = (
        subcrop_xy - hr_offset_x + int(round(lr_cropped.shape[2] * scale_factor_xy))
    )

    lr_z_start = 0
    lr_z_stop = lr_cropped.shape[0]
    lr_y_start = 0
    lr_y_stop = lr_cropped.shape[1]
    lr_x_start = 0
    lr_x_stop = lr_cropped.shape[2]

    if hr_z_start < 0:
        lr_z_start += abs(int(round(hr_z_start / scale_factor_z)))
        hr_z_start = 0
    if hr_z_stop > hr.shape[0]:
        lr_z_stop -= int(round((hr_z_stop - hr.shape[0]) / scale_factor_z))
        hr_z_stop = hr.shape[0]
    if hr_y_start < 0:
        lr_y_start += abs(int(round(hr_y_start / scale_factor_xy)))
        hr_y_start = 0
    if hr_y_stop > hr.shape[1]:
        lr_y_stop -= int(round((hr_y_stop - hr.shape[1]) / scale_factor_xy))
        hr_y_stop = hr.shape[1]
    if hr_y_start < 0:
        lr_x_start += abs(int(round(hr_x_start / scale_factor_xy)))
        hr_x_start = 0
    if hr_x_stop > hr.shape[2]:
        lr_x_start -= int(round((hr_x_stop - hr.shape[2]) / scale_factor_xy))
        hr_x_stop = hr.shape[2]

    lr_final = lr_cropped[
        lr_z_start:lr_z_stop, lr_y_start:lr_y_stop, lr_x_start:lr_x_stop
    ]
    hr_final = hr[hr_z_start:hr_z_stop, hr_y_start:hr_y_stop, hr_x_start:hr_x_stop]


    hr_true_shape = (
        int(round(lr_final.shape[0] * scale_factor_z)),
        int(round(lr_final.shape[1] * scale_factor_xy)),
        int(round(lr_final.shape[2] * scale_factor_xy)),
    )

    scale_ratios = np.divide(
        np.array(hr_final.shape, dtype=np.float),
        np.array(lr_final.shape, dtype=np.float),
        dtype=np.float,
    )

    error = np.array(hr_final.shape, dtype=np.int) - np.array(
        hr_true_shape, dtype=np.int
    )

    lr_z = [lr_z_start + n_z, lr_z_stop + n_z]
    lr_x = [lr_x_start + n_xy, lr_x_stop + n_xy]
    lr_y = [lr_y_start + n_xy, lr_y_stop + n_xy]
    lr_crops = [lr_z, lr_y, lr_x]

    hr_z = [hr_z_start, hr_z_stop]
    hr_x = [hr_x_start, hr_x_stop]
    hr_y = [hr_y_start, hr_y_stop]
    hr_crops = [hr_z, hr_y, hr_x]

    if np.any(np.absolute(error) > 0):


        if hr_z_stop - error[0] <= hr.shape[0]:
            hr_z_stop = hr_z_stop - error[0]
        elif hr_z_start + error[0] >= 0:
            hr_z_start = hr_z_start + error[0]
        if hr_y_stop - error[1] <= hr.shape[1]:
            hr_y_stop = hr_y_stop - error[1]
        elif hr_y_start + error[1] >= 0:
            hr_y_start = hr_y_start + error[1]
        if hr_x_stop - error[2] <= hr.shape[2]:
            hr_x_stop = hr_x_stop - error[2]
        elif hr_x_start + error[2] >= 0:
            hr_x_start = hr_x_start + error[2]

        hr_z = [hr_z_start, hr_z_stop]
        hr_x = [hr_x_start, hr_x_stop]
        hr_y = [hr_y_start, hr_y_stop]

        hr_crops = [hr_z, hr_y, hr_x]
        hr_final = hr[hr_z[0] : hr_z[1], hr_y[0] : hr_y[1], hr_x[0] : hr_x[1]]
        error = np.array(hr_final.shape, dtype=np.int) - np.array(
            hr_true_shape, dtype=np.int
        )
        if np.any(np.absolute(error) > 0):
            return None, None


    return lr_crops, hr_crops


