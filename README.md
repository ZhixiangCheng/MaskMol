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

