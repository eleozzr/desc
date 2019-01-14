from setuptools import setup, find_packages

setup(name = 'desc',
			version = '0.1.4.0',
			description = 'Deep Embedded Single cell rna-seq Clustering',
			long_description = 'DESC is an unsupervised deep learning algorithm for clustering scRNA-seq data. The algorithm construct a non-linear mapping function from the original scRNA-seq data space to a low-dimensional feature space by iteratively learns cluster-specific gene expression representation and cluster assignments based on a deep neural network. This iterative procedure moves each cell to its nearest cluster, balances biological and technical differences between clusters, and reduces the influence of batch effect. DESC also enables soft clustering by assigning cluster-specific probabilities to each cell, which facilitates the identification of cells clustered with high-confidence and interpretation of results.',
			classifiers = [
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
      	],
			url = 'https://yafei611.github.io/desc/',
			author = 'Xiangjie Li, Gang Hu, Mingyao Li, Yafei Lyu',
			author_email = 'ele717@163.com, huggs@nankai.edu.cn, mingyao@pennmedicine.upenn.edu, lyuyafei@gmail.com',
			license = 'MIT',
			packages = find_packages(),
			include_package_data=True,
			install_requires = [
				'pydot', 'python-igraph', 'tensorflow',
				'keras', 'scanpy', 'pandas', 'louvain'
				],
			zip_safe = False
			)
			