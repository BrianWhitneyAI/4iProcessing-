## Running Snakemake workflow

This step assumes you have run up to step 2(matched dataframe creation)

To run snakemake:

cd /multiplex_immuno_processing/multiplex_immuno_processing/batch_processing

snakemake --cores 1 -p --profile {} --forceall --configfile {}

--profile specifies the a yaml file that contains hardware specific requirements for the system that you are running on

--configfile specifies a config file that contains some basic information for the run and specifies the location of the yaml config file created in step 1(create an initial config file)

--forceall - optional; this force runs all steps in the dag
