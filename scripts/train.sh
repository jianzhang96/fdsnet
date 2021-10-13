CUDA_VISIBLE_DEVICES=0 python train.py --model fdsnet --use-ohem True --aux True \
									   --dataset phone_voc --lr 0.0001 --epochs 150 \
									   --batch-size 8