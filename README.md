# MaskMol

## Install environment

**1. GPU environmentx**<br>  
CUDA 11.1

**2. create a new conda environment**<br>  
conda create -n MaskMol python=3.7<br>  
conda activate MaskMol  

**3. download some packages**<br>  
pip install -r requirements.txt<br>  
source activate MaskMol  

## Pretraining
Download [pretraining data](https://drive.google.com/file/d/1VGCADtln1NRswoOnnGa9iWyM2Xe9gmq7/view?usp=sharing) and put it into ./datasets/pretrain/<br>  
**1. get masking image**<br>  
```
python ./data_process/mask_parallel.py --jobs 15
```
**Note:** You can find the img, Atom, Bond, and Morif in datasets/pretrain<br>  

**2. lmdb process**<br>  
```
python ./data_process/lmdb_process.py --jobs 15
```
**Note:** You can find the four files (img_lmdb, Atom_lmdb, Bond_lmdb, Motif_lmdb) in datasets/pretrain<br>  

**3. start to pretrain**<br>  
Usage:<br>  
```
usage: train_muti_GPU_lmdb.py [-h] [--lr LR] [--lrf LRF] [--nums NUMS]
                              [--wd WD] [--workers WORKERS]
                              [--val_workers VAL_WORKERS] [--epochs EPOCHS]
                              [--start_epoch START_EPOCH] [--batch BATCH]
                              [--momentum MOMENTUM]
                              [--checkpoints CHECKPOINTS] [--resume PATH]
                              [--seed SEED] [--data_path DATA_PATH]
                              [--log_dir LOG_DIR] [--proportion PROPORTION]
                              [--ckpt_dir CKPT_DIR] [--verbose] [--ngpu NGPU]
                              [--gpu GPU] [--Atom_lambda ATOM_LAMBDA]
                              [--Bond_lambda BOND_LAMBDA]
                              [--Motif_lambda MOTIF_LAMBDA] [--nodes NODES]
                              [--ngpus_per_node NGPUS_PER_NODE]
                              [--dist-url DIST_URL] [--node_rank NODE_RANK]
```
Code to pretrain:<br>  
```
python train_muti_GPU_lmdb.py --nodes 1 \
                   --ngpus_per_node 4 \
                   --gpu 0,1,2,3 \
                   --batch 128 \
                   --epochs 50 \
                   --proportion 0.5 \
                   --Atom_lambda 1 \
                   --Bond_lambda 1 \
                   --Motif_lambda 1 \
                   --nums 2000000
```
For testing, you can simply pre-train MaskMol using single GPU on 20w dataset:
```
python train_lmdb.py --gpu 0 \
                   --batch 32 \
                   --epochs 50 \
                   --proportion 0.5 \
                   --Atom_lambda 1 \
                   --Bond_lambda 1 \
                   --Motif_lambda 1 \
                   --nums 200000
```

## Finetuning

**1. Download pre-trained ImageMol** <br>  
You can download pre-trained model and push it into the folder ckpts/ <br> 

**2. Finetune with pre-trained MaskMol** <br>  
a) You can download [activity cliffs estimation](https://github.com/molML/MoleculeACE/tree/main/MoleculeACE/Data/benchmark_data) and [compound potency prediction](https://github.com/TiagoJanela/ML-for-compound-potency-prediction/tree/main/dataset) put it into datasets/finetuning/ <br> 

b) The usage is as follows:
```
usage: finetuning.py [-h] [--dataset DATASET] [--dataroot DATAROOT]
                     [--gpu GPU] [--ngpu NGPU] [--workers WORKERS] [--lr LR]
                     [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM]
                     [--seed SEED] [--runseed RUNSEED] [--epochs EPOCHS]
                     [--start_epoch START_EPOCH] [--batch BATCH]
                     [--resume PATH] [--imageSize IMAGESIZE] [--image_aug]
                     [--save_finetune_ckpt {0,1}] [--eval_metric EVAL_METRIC]
                     [--log_dir LOG_DIR] [--ckpt_dir CKPT_DIR]
```
For example:
```
python finetuning_cliffs.py --gpu 0 \
                   --save_finetune_ckpt 1 \
                   --dataroot ./datasets/finetuning/cliffs \
                   --dataset CHEMBL219_Ki \
                   --resume ./ckpts/pretrain/MaskMol.pth.tar \
                   --lr 5e-4 \
                   --batch 16 \
                   --epochs 100 \
                   --eval_metric rmse
```

## Attention Visualization
More about GradCAM heatmap can be found from this link: [https://github.com/jacobgil/vit-explain](https://github.com/jacobgil/vit-explain) <br> 

We also provide a script to generate GradCAM heatmaps:

```
usage: vit_explain.py [-h] [--use_cuda] [--image_path IMAGE_PATH]
                      [--molecule_path MOLECULE_PATH]
                      [--head_fusion HEAD_FUSION]
                      [--discard_ratio DISCARD_RATIO]
                      [--category_index CATEGORY_INDEX] [--resume PATH]
                      [--gpu GPU]
```
you can run the following script:

```
python main.py --resume MaskMol \
               --img_path 1.png \
               --discard_ratio 0.9 \
               --gpu 0 \
               --category_index 0
```











