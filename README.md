## NetworksPipeline
Pipeline for finding the optimal edgeweight cut-off for genome-genome k-mer similarity networks.

# How it works
This script takes a Sourmash (https://github.com/sourmash-bio/sourmash) csv output matrix and converts it to an edgetable for use in cytoscape. 
It then uses iGraph Louvain method to iteratively check for communities with this edgetable, removing edges below a cutoff, then checking again. 
The number of communities at each iteration is counted, and the 'elbow' or 'knee' of the curve calculated using kneebow. 

This 'knee' is then taken as the optimal edgeweight cut off for the network. With the idea that cutoffs below are too interconnected and cutoffs above are too fragmented.
