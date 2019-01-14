
## Deep Embedding Single-cell Clustering 

DESC is an unsupervised deep learning algorithm for clustering scRNA-seq data. The algorithm construct a non-linear mapping function from the original scRNA-seq data space to a low-dimensional feature space by iteratively learns cluster-specific gene expression representation and cluster assignments based on a deep neural network. This iterative procedure moves each cell to its nearest cluster, balances biological and technical differences between clusters, and reduces the influence of batch effect. DESC also enables soft clustering by assigning cluster-specific probabilities to each cell, which facilitates the identification of cells clustered with high-confidence and interpretation of results. 

![DESC workflow](docs/assets/images/desc_workflow.png)

For thorough details, see the preprint: [Bioxiv](https://www.biorxiv.org)
<br>

## Usages

The [**desc**](https://github.com/Yafei611/desc) package is an implementation of deep embedding single-cell clustering. With desc, you can:

- Preprocess single cell expression data from varies of formats
- Build low-dimensional representation of single cell expression
- Obtain a soft-clustering assignments of cells
- Visualize cell clusters separation and expression pattern
<br>

## Installation

To install the current release:

```
pip install desc
```

Please check DESC [Tutorial](https://yafei611.github.io/desc/tutorial.html) for more detials.
<br>

## Contributing

Souce code: [Github](https://github.com/Yafei611/desc)  

We are continuing to add new features. Any kind of contribution, like bug reports or feature requests, are welcome.
<br>

## References

Please consider to cite the references in case they are used.  
[1] DESC    
[2] DESC
<br>

## Licence

Copyright (C) 2018 Mingyao's Lab

