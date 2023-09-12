# multiplex_immuno_processing

This workflow is used to align images across different rounds of imaging on data acquired on ZSD microscopes. This workflow can be run in several ways. (1) In a manual sequence where we run a set of python scripts one after another, (2) In a hybrid sequential/parrelilized setup where we can parralize the most time consuming parts of this workflow across different nodes via slurm, (3) In a modular setup where we can perform round alignment for just a specific desired position, (4) Snakemake?

This workflow has 4 main steps:

## Create an intial yaml configuration file 
This yaml specifies what data can be used for round alignment. Examples of yaml configuration files are provided in config_files/
We also provide a script that can generate an initial config which can then be manually modified. This may or may not work depending on the folder and naming structure of the input data

python multiplex_immuno_processing/generate_initial_config.py --input_dirs {} --output_dirs {}

## Generate matched position dataframes
This specifies which position/scenes correspond to which position/scenes across different rounds of imaging

```
python multiplex_immuno_processing/find_matched_positions_across_rounds.py --input_yaml {} --refrence_round {}
```

The output of this is a directory called matched_datasets which contains a csv for each position with the corresponding position/scenes across each round

<img width="514" alt="Screenshot 2023-09-12 at 3 57 39 PM" src="https://github.com/aics-int/multiplex_immuno_processing/assets/40441855/0192c323-78d3-4276-8200-bc415c594d5f">

## Find alignment parameters
This step finds the alignment parameters for each position in the matched position dataframe

```
python multiplex_immuno_processing/find_alignment_parameters.py --input_yaml ../../config_files/{} --matched_position_csv_dir {}
```

The argument matched_position_csv_dir is optional and if specified, this will only run the single position that is desired, otherwise this will find the alignment parameters for each position sequentially

or to parrallize on slurm:

```
cd multiplex_immuno_processing/batch_processing
python tempelate_batch_run.py --input_yaml ../../config_files/{} --matched_position_csv_parent_dir {}
```
The output of this is a directory called alignment_parameters which contains csvs for each position with the calculated alignment parameters

<img width="584" alt="Screenshot 2023-09-12 at 3 58 03 PM" src="https://github.com/aics-int/multiplex_immuno_processing/assets/40441855/7dbdb2b1-7ba7-402c-b721-7d2c154df6bc">


## Apply alignment

This step performs a rigid registration according to the alignment parameters calculated in the previous step

```
python multiplex_immuno_processing/apply_registration.py --input_yaml ../../config_files/{} --matched_position_w_align_params_csv {}
```

The argument matched_position_w_align_params_csv is optional and if specified, this will only run the single position that is desired, otherwise this will allign each position sequentially

or to parrallize on slurm:

```
cd multiplex_immuno_processing/batch_processing
python tempelate_batch_run.py --input_yaml ../../config_files/{} --matched_position_w_align_params_csv_parent_dir {}
```

The output of this is a directory called round_aligned_images which contains all the aligned positions for each round

# Gif overlay validation(optional):
This step creates gifs that show overlays for each position. This can be used for quick validation of the alignments
```
python multiplex_immuno_processing/generate_gifs_for_validation.py --input_yaml {} --frame_rate {}
```
![3500005822_position_05_evaluation](https://github.com/aics-int/multiplex_immuno_processing/assets/40441855/07868274-cb75-42c3-a554-06a335c0c2b6)







