# multiplex_immuno_processing

4i Processing contains programatic steps for imaging processing 

These are instuctions for reproducing last year's 4i data for plates 724, 725, 726, and 733. Keeping for now for historical purposes just in case

# how to run this code

### get initial configs automatically
```
python ./multiplex_immuno_processing/generate_initial_config.py --output_path "//allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports" --input_dirs  "//allen/aics/microscopy/Antoine/Analyse EMT/4i Data/5500000733" "//allen/aics/microscopy/Antoine/Analyse EMT/4i Data/5500000724" "//allen/aics/microscopy/Antoine/Analyse EMT/4i Data/5500000728" "//allen/aics/microscopy/Antoine/Analyse EMT/4i Data/5500000726" "//allen/aics/microscopy/Antoine/Analyse EMT/4i Data/5500000725"
```

### manually edit config files to ensure that information is correct
 be sure to check round 1 of 5500000725 to remove this file: `"\\allen\aics\microscopy\Antoine\Analyse EMT\4i Data\5500000725\ZSD2\Round 1\5500000725_20X_first-scene.czi"`

### get parent image metadata for images associated with each barcode
```
python ./multiplex_immuno_processing/gather_parent_image_metadata.py  --output_path "//allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports"
```

### check the parent image metadata to determine which positions are present less or more than once per round of imaging
```
python ./multiplex_immuno_processing/check_parent_image_metadata_dataframe.py  --output_path "//allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports"
```

### use config and metadata dataframe to match all positions across rounds of imaging for each barcode
```
python ./multiplex_immuno_processing/generate_matched_position_dataframe.py  --output_path "//allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports"
```


### and then filter out the positions with multiple matches
```
python ./multiplex_immuno_processing/filter_matched_position_dataframe_to_remove_multiple_matches.py  --output_path "//allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports"
```

### check the position matching result
```
python ./multiplex_immuno_processing/check_matched_position_dataframe.py  --output_path "//allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports"
```

### now compute alignment paramters based on which method is prefered

```
python .\multiplex_immuno_processing\generate_alignment_parameters_tempelate.py --output_path "output_path" --barcode "barcode" --method "method to use" --position_list(optional)

```

### next , check the alignment parameter computation result

python ./multiplex_immuno_processing/check_alignment_parameters_dataframe.py  --output_path "//allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports" --method "cross_cor"

### next execute the alignment
#### Note that the jinja tempelate should also be modified here to specify your own output directory for the logs

```
python .\multiplex_immuno_processing\round_alignment_tempelate.py --output_path "output_path" --barcode "barcode" --method "method to use" --position_list(optional)
```

### next evaluate the alignment
```
python .\multiplex_immuno_processing\generate_contact_sheet_gif.py  --input_dir "dir pointing to mip outputs" --output_dir "output directory to save gifs" --barcode "barcode(int)" --frame_rate "frame rate of gif(int)"
```
#### To address problems in the alignment that can be solved by using different alignment methods for different rounds:

```
python multiplex_immuno_processing/create_merged_aligned_params.py --barcode {} --output_path {} --output_dir_merged {} --position_list_to_use_ORB {} 
```

This scripts lets us use the ORB alignment for round 1 on certain positions and keep the cross_corr alignment for everything else. This script outputs csv files with the best_alignment_params column that will have which parameters to use

Then we can run round_alignment_tempelate.py again specifiying using the merged column

# to run on slurm use
srun -p aics_cpu_general --mem 70G --pty bash #
module load anaconda3
source activate frick_multiplex_test
cd //allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4iProcessing-/
