
## Deep Embedding Single-cell Clustering 

DESC is an unsupervised deep learning algorithm for clustering scRNA-seq data. The algorithm construct a non-linear mapping function from the original scRNA-seq data space to a low-dimensional feature space by iteratively learns cluster-specific gene expression representation and cluster assignments based on a deep neural network. This iterative procedure moves each cell to its nearest cluster, balances biological and technical differences between clusters, and reduces the influence of batch effect. DESC also enables soft clustering by assigning cluster-specific probabilities to each cell, which facilitates the identification of cells clustered with high-confidence and interpretation of results. 

![DESC workflow](docs/assets/images/desc_workflow.png)

For thorough details, see the preprint:
https://www.biorxiv.org

## Usages

The **desc** package is an implementation of deep embedding single-cell clustering. With desc, you can:

- Preprocess single cell expression data from varies of formats
- Build low-dimensional representation of single cell expression
- Obtain a soft-clustering assignments of cells
- Visualize cell clusters separation and expression pattern

## Installation

To install the current release:

```
pip install desc
```

## Contributing

Souce code: https://github.com/Yafei611

We are continuing to add new features. Any kind of contribution, like bug reports or feature requests, are welcome.

## References

Please cite the references appropriately in case they are used.  
[1] DESC    
[2] DESC


## Licence

Copyright (C) 2018 Mingyao's Lab

