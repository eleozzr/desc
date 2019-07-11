|PyPI| |bioconda| |Docs| |Build Status|

.. |PyPI| image:: https://img.shields.io/pypi/v/scanpy.svg
   :target: https://pypi.org/project/scanpy
.. |bioconda| image:: https://img.shields.io/badge/bioconda-🐍-blue.svg
   :target: http://bioconda.github.io/recipes/scanpy/README.html
.. |Docs| image:: https://readthedocs.com/projects/icb-scanpy/badge/?version=latest
   :target: https://scanpy.readthedocs.io
.. |Build Status| image:: https://travis-ci.org/theislab/scanpy.svg?branch=master
   :target: https://travis-ci.org/theislab/scanpy
..
   .. |Coverage| image:: https://codecov.io/gh/theislab/scanpy/branch/master/graph/badge.svg
      :target: https://codecov.io/gh/theislab/scanpy

DESC--Deep Embedded Single-cell RNA-seq Clustering
=======================================
DESC is an unsupervised deep learning algorithm for clustering scRNA-seq data. The algorithm constructs a non-linear mapping function from the original scRNA-seq data space to a low-dimensional feature space by iteratively learning cluster-specific gene expression representation and cluster assignment based on a deep neural network. This iterative procedure moves each cell to its nearest cluster, balances biological and technical differences between clusters, and reduces the influence of batch effect. DESC also enables soft clustering by assigning cluster-specific probabilities to each cell, which facilitates the identification of cells clustered with high-confidence and interpretation of results.


Read the tutorial_.

.. _tutorial: https://eleozzr.github.io/desc/

