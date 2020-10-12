prepare:
	mkdir pretrained
	wget -O pretrained/PANNsCNN14Att.pth https://zenodo.org/api/files/85f997fd-fdf1-4df0-930f-a14c476cef8d/Cnn14_DecisionLevelAtt_mAP%3D0.425.pth
	make -C input/birdsong-recognition

train:
	make train-stage1
	make train-stage2
	make train-stage3

train-stage1:
	CUDA_VISIBLE_DEVICES=0 python train.py --config configs/000_PANNs_stage1.yml
	CUDA_VISIBLE_DEVICES=0 python sed.py --config configs/sed/000_Stage1_sed.yml
	python find_missing_label.py

train-stage2:
	make train-stage2-v1
	make train-stage2-v2

train-stage3:
	make train-stage3-no-ext
	make train-stage3-ext

train-stage2-v1:
	make train-stage2-v1-no-ext
	make train-stage2-v1-ext

train-stage2-v2:
	make train-stage2-v2-no-ext
	make train-stage2-v2-ext

train-stage2-v1-no-ext:
	CUDA_VISIBLE_DEVICES=0 python train.py --config configs/001_ResNestSED_stage2_v1.yml
	CUDA_VISIBLE_DEVICES=0 python sed_soft.py --config configs/sed/001_Stage2_sed_v1.yml

train-stage2-v1-ext:
	CUDA_VISIBLE_DEVICES=0 python sed_soft.py --config configs/sed/003_Stage2_sed_extended_v1.yml

train-stage2-v2-no-ext:
	CUDA_VISIBLE_DEVICES=0 python train.py --config configs/002_ResNestSED_stage2_v2.yml
	CUDA_VISIBLE_DEVICES=0 python sed_soft.py --config configs/sed/002_Stage2_sed_v2.yml

train-stage2-v2-ext:
	CUDA_VISIBLE_DEVICES=0 python sed_soft.py --config configs/sed/004_Stage2_sed_extended_v2.yml

train-stage3-no-ext:
	CUDA_VISIBLE_DEVICES=0 python ema.py --config configs/003_ResNestSED_EMA_stage3_v1.yml
	CUDA_VISIBLE_DEVICES=0 python ema.py --config configs/004_ResNestSED_EMA_th04_stage3_v1.yml
	CUDA_VISIBLE_DEVICES=0 python ema.py --config configs/005_ResNestSED_EMA_th06_stage3_v1.yml
	CUDA_VISIBLE_DEVICES=0 python ema.py --config configs/006_ResNestSED_EMA_stage3_v2.yml
	CUDA_VISIBLE_DEVICES=0 python ema.py --config configs/007_ResNestSED_EMA_th03_stage3_v2.yml
	CUDA_VISIBLE_DEVICES=0 python ema.py --config configs/008_ResNestSED_EMA_th04_stage3_v2.yml
	CUDA_VISIBLE_DEVICES=0 python ema.py --config configs/009_ResNestSED_EMA_th06_stage3_v2.yml
	CUDA_VISIBLE_DEVICES=0 python ema.py --config configs/010_ResNestSED_EMA_th07_stage3_v2.yml
	CUDA_VISIBLE_DEVICES=0 python ema.py --config configs/012_EfficientNetSED_EMA_stage3_v2.yml
	CUDA_VISIBLE_DEVICES=0 python ema.py --config configs/013_EfficientNetSED_EMA_th04_stage3_v2.yml

train-stage3-ext:
	CUDA_VISIBLE_DEVICES=0 python ema.py --config configs/011_ResNestSED_EMA_ext_stage3_v2.yml
