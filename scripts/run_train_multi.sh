CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
-cfg='configs/pit_ti.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='datasets/ImageNet1K' \
-save_path='output'
