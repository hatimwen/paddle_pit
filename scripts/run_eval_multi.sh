CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
-cfg='./configs/pit_ti.yaml' \
-dataset='imagenet2012' \
-batch_size=128 \
-data_path='datasets/ImageNet1K' \
-eval \
-pretrained='output/Best_PiT'
