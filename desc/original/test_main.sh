#!/bin/bath
#no any filter and directly use data
#baron_mouse
python3  __main__.py --input /data0/xiangjie/silverstandardData/Baron_mouse \
	--prefilter_cells True \
	--filter_cells_mito False \
	--prefilter_gene True \
	--normalize_per_cell True \
	--find_variable_genes True \
	--log1p False \
	--scale_data 1 \
	--max_value 4 \
	--louvain_resolution 0.4 0.6 0.8 \
	--batch_size 256 \
	--save_dir result_tmp2;

		
