from pathlib import Path
import xml.etree.ElementTree as ET

from aicsimageio import AICSImage
import lxml.etree as etree
import numpy as np
import pandas as pd


def compute_adjusted_xy(df, overwrite=True):
    if "X_original" not in df.columns.tolist():
        for xi, xyz in enumerate(["X", "Y"]):
            xyz0 = df[xyz].tolist()
            df[xyz + "_original"] = xyz0
            xyzadjust = []
            anchorlist = df["PlateAnchorPoint"].tolist()
            for i in range(df.shape[0]):
                xyzadjust.append(xyz0[i] - anchorlist[i][xi])
            df[xyz + "_adjusted"] = xyzadjust
            if overwrite:
                df[xyz] = xyzadjust
    else:
        print("already adjsuted")
    return df


def get_position_info_from_czi(filename):

    reader = AICSImage(filename)
    meta0 = reader.metadata
    # convert metadata to lxml
    metastr = ET.tostring(meta0).decode("utf-8")
    meta = etree.fromstring(metastr)

    """
        info_attrib_list = [
            "Name",
            "X",
            "Y",
            "Z",
            "Id",
            "IsUsedForAcquisition",
            "imgsize_um",
            "imgsize_pixels",
            "pixelSizes",
            "CameraPixelAccuracy",
            "parent_file",
        ]
    """

    feats = {}  # define a dictionary for recording the extracted metadata

    # record the filename
    feats["file"] = filename
    feats["parent_file"] = filename

    # find the image dimensions of the image
    # number of scenes
    number_of_scenes_acquired = eval(meta.find(".//SizeS").text)

    # number of z slices per image
    SizeZ = eval(meta.find(".//SizeZ").text)

    # get camera dimensions
    ImageFrameAll = list(meta.findall(".//ParameterCollection/ImageFrame"))
    frame_size_pixels = eval(ImageFrameAll[0].text)

    # number of pixels in each dimension for a given scene (size in X,Y,Z)
    feats["imgsize_pixels"] = tuple(
        (frame_size_pixels[-2], frame_size_pixels[-1], SizeZ)
    )

    # find key imaging parameters
    ImagePixelDistancesList = meta.findall(".//ParameterCollection/ImagePixelDistances")

    for ip in ImagePixelDistancesList[0:1]:  # only choose the first camera
        feats["ImagePixelDistances"] = tuple(eval(ip.text))

        feats["totalmagnification"] = eval(
            ip.getparent().find("./TotalMagnification").text
        )

        feats["CameraPixelAccuracy"] = eval(
            ip.getparent().find("./CameraPixelAccuracy").text
        )

    channels = meta.findall(".//Information/Image/Dimensions/Channels/Channel")
    channel_dict = {"channel_indices": [], "channel_names": []}
    for channel in channels:
        index = channel.attrib["Id"].replace("Channel:", "")
        channel_name = channel.attrib["Name"]
        channel_dict["channel_indices"].append(index)
        channel_dict["channel_names"].append(channel_name)
        exposure_time = int(channel.find(".//ExposureTime").text) / 1e6
        binning = channel.find(".//Binning").text

        try:
            laser_wavelength = channel.find(".//ExcitationWavelength").text
            laser_intensity = channel.find(".//Intensity").text
        # except ValueError("Some Value Error") as e:
        except Exception as e:
            print(str(e))
            laser_wavelength = "None"
            laser_intensity = "None"

        feats["laser_wavelength" + "_" + channel_name] = laser_wavelength
        feats["exposure_time" + "_" + channel_name] = exposure_time
        feats["binning"] = binning
        feats["laser_intensity" + "_" + channel_name] = laser_intensity
    feats["channel_dict"] = str(channel_dict)

    ZStepfind = meta.find(".//Z/Positions/Interval/Increment")
    ZStep = eval(ZStepfind.text)
    xypxsize = np.asarray(feats["ImagePixelDistances"]) / feats["totalmagnification"]
    feats["pixelSizes"] = (xypxsize[0], xypxsize[1], ZStep)  # units of um
    feats["imgsize_um"] = tuple(
        [x * y for x, y in zip(feats["pixelSizes"], feats["imgsize_pixels"])]
    )
    feats["PlateAnchorPoint"] = eval(
        "[" + meta.find(".//Template/AnchorPoint").text + "]"
    )
    feats["PlateReferencePoint"] = eval(
        "[" + meta.find(".//Template/ReferencePoint").text + "]"
    )

    dfmetalist = []
    for regions in [meta.find(".//SingleTileRegions")]:
        # some weird czis have duplicates in SingleTileRegions....
        # so you need to drop those by not doing find all
        for region in regions.findall("SingleTileRegion"):

            attrib = region.attrib
            feats["Name"] = attrib["Name"]

            for info in region.findall("X"):
                feats["X"] = float(info.text)
            for info in region.findall("Y"):
                feats["Y"] = float(info.text)
            for info in region.findall("Z"):
                feats["Z"] = float(info.text)
            for info in region.findall("IsUsedForAcquisition"):
                feats["IsUsedForAcquisition"] = info.text

            dfmetalist.append(pd.DataFrame(data=feats.values(), index=feats.keys()).T)
    df1 = pd.concat(dfmetalist)

    # now search for next set of parameters from .//Scene xml region
    # this pulls out the differences between how scene number and position number!
    dfsp = []

    for region in meta.findall(".//Scene"):
        feats = {}
        attrib = region.attrib
        feats["Position"] = attrib["Name"]
        feats["Position_num"] = int(feats["Position"].replace("P", ""))
        feats["Scene"] = int(attrib["Index"]) + 1
        subregion = region.find("Shape")
        feats["Well_id"] = "unknown"  # in case Shape is not a feature
        for subregion in region.findall("Shape"):
            feats["Well_id"] = subregion.attrib["Name"]
        dfsp.append(pd.DataFrame(data=feats.values(), index=feats.keys()).T)

    df2 = pd.concat(dfsp)
    df = pd.merge(df1, df2, left_on="Name", right_on="Position", suffixes=("_1", "_2"))
    dfsub = df[df["IsUsedForAcquisition"] == "true"]
    dfsub = dfsub[dfsub["Scene"].astype(int) <= number_of_scenes_acquired]
    dfsub["fname"] = dfsub["file"].apply(lambda x: Path(x).stem)
    dfsub = compute_adjusted_xy(dfsub)

    return dfsub
