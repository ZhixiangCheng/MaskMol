# MaskMol

## Install environment

1. **GPU environmentx**<br>  
CUDA 11.1

2. **create a new conda environment**<br>  
conda create -n MaskMol python=3.7<br>  
conda activate MaskMol  

3. **download some packages**<br>  
pip install -r requirements.txt<br>  
source activate MaskMol  

## Pretraining
1. **get masking image**
''' python mask_parallel.py --jobs 15 '''

3. **lmdb process**  
