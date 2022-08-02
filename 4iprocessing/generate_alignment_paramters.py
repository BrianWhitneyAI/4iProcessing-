# this code computes the alignment parameters to align each scene across the multiple rounds of imaging
# should take a barcode as an argument,
# then this code reads that dataframe pickle for that given barcode
# then it uses the dataframe to determine the reference channel to be used for each of the rounds for alignment
# then it loads all the reference channel images for each round
# then it runs the an alignment algorithm to align all of the rounds 
## all positions should be aligned to round 1 (that will be the reference round)
## this will enable all the processing to be run in stages later on as new data is acquired
