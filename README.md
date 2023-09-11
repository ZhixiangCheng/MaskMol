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
python train_muti_GPU_lmdb.py
                   --gpu 0 \
                   --batch 32 \
                   --epochs 50 \
                   --proportion 0.5 \
                   --Atom_lambda 1 \
                   --Bond_lambda 1 \
                   --Motif_lambda 1 \
                   --nums 200000
```
