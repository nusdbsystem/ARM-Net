# Example Script for Running Benchmark Datasets

# ARM-Net

## Frappe
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet  --alpha 2.0  --h  32  --nattn_head  8 --lr 0.001  --exp_name frappe_armnet
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet  --ensemble  --alpha 1.5 --h 4 --nattn_head 4  --lr 0.003  --exp_name frappe_armnet+

##  MovieLens
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet --h 16  --nattn_head 1   --alpha 2.0 --lr 0.001  --exp_name ML_armnet  --nfield 3 --nfeat 92000 --dataset movielens
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet  --ensemble  --alpha 2.5  --h  8 --nattn_head 1  --lr 0.001  --exp_name ML_armnet+ --nfield 3 --nfeat 92000 --dataset movielens

##  Avazu
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet --nlayer 3 --mlp_hid 200 --h  32  --nattn_head 1   --alpha 1.5 --lr 0.001  --exp_name AV_armnet  --nfield 22 --nfeat 1600000 --dataset avazu --eval_freq 1000
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet  --nlayer 3 --mlp_hid 200 --dnn_nlayer 3 --dnn_hid 200  --ensemble  --alpha 2.0  --h  8  --nattn_head 8   --lr 0.001 --exp_name AV_armnet+  --nfield 22 --nfeat 1600000 --dataset avazu --eval_freq 1000

##  Criteo
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet --nlayer 2 --mlp_hid 500 --h  64  --nattn_head 4  --alpha 2.0 --lr 0.001 --exp_name CR_armnet  --nfield 39 --nfeat 2100000 --dataset criteo
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet  --mlp_hid 500 --nlayer 2 --dnn_nlayer 2 --dnn_hid 500  --ensemble  --alpha 2.0  --h  32   --nattn_head 4  --lr 0.003   --exp_name CR_armnet+  --nfield 39 --nfeat 2100000 --dataset criteo

##  Diabetes130
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet --nlayer 1 --h  1  --nattn_head 32   --alpha 1.7 --lr 0.003  --batch_size 1024  --exp_name DB_armnet  --nfield 43 --nfeat 369 --dataset uci_diabetes
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet  --nlayer 1 --dnn_nlayer 1  --ensemble  --alpha 1.5  --h  64   --nattn_head 8    --lr 0.01 --batch_size 1024  --exp_name DB_armnet+  --nfield 43 --nfeat 369 --dataset uci_diabetes





# ARM-Net w/o bilinear-weight to reduce one hyper-param (nhead)

## Frappe
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet_1h  --alpha 2.0  --h  128  --lr 0.001  --exp_name frappe_armnet1h
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet_1h  --ensemble  --alpha 1.5 --h 128  --lr 0.003  --exp_name frappe_armnet1h+

##  MovieLens
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet_1h --h 128  --alpha 2.0 --lr 0.001  --exp_name ML_armnet1h  --nfield 3 --nfeat 92000 --dataset movielens
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet_1h  --ensemble  --alpha 2.5  --h  128  --lr 0.001  --exp_name ML_armnet1h+ --nfield 3 --nfeat 92000 --dataset movielens

##  Avazu
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet_1h --nlayer 3 --mlp_hid 200 --h  128   --alpha 1.5 --lr 0.001  --exp_name AV_armnet1h  --nfield 22 --nfeat 1600000 --dataset avazu --eval_freq 1000
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet_1h  --nlayer 3 --mlp_hid 200 --dnn_nlayer 3 --dnn_hid 200  --ensemble  --alpha 2.0  --h  128   --lr 0.001 --exp_name AV_armnet1h+  --nfield 22 --nfeat 1600000 --dataset avazu --eval_freq 1000

##  Criteo
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet_1h --nlayer 2 --mlp_hid 500 --h  128  --alpha 2.0 --lr 0.001 --exp_name CR_armnet1h  --nfield 39 --nfeat 2100000 --dataset criteo
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet_1h  --mlp_hid 500 --nlayer 2 --dnn_nlayer 2 --dnn_hid 500  --ensemble  --alpha 2.0  --h  128  --lr 0.003   --exp_name CR_armnet1h+  --nfield 39 --nfeat 2100000 --dataset criteo

##  Diabetes130
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet_1h --nlayer 1 --h  128  --alpha 1.7 --lr 0.003  --batch_size 1024  --exp_name DB_armnet1h  --nfield 43 --nfeat 369 --dataset uci_diabetes
CUDA_VISIBLE_DEVICES=0 python train.py --model armnet_1h  --nlayer 1 --dnn_nlayer 1  --ensemble  --alpha 1.5  --h  128    --lr 0.01 --batch_size 1024  --exp_name DB_armnet1h+  --nfield 43 --nfeat 369 --dataset uci_diabetes
