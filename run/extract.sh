python -u HyperspecAE/src/extract.py\
	-src_dir HyperspecAE/data/data.mat\
	-ckpt HyperspecAE/logs/final_model.pt\
	-num_bands 3\
	-end_members 2\
	-encoder_type deep\
	-soft_threshold SReLU\
	-activation Leaky-ReLU\
	-gaussian_dropout 0.2\
	-threshold 1.0
