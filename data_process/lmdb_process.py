import argparse
import os
import lmdb
import pickle
from PIL import Image
from io import BytesIO
from tqdm import *
import time
from joblib import Parallel, delayed


def get_atom_bond_motif_label(file, index):
        
    d = os.listdir(file)
    f = [file + "/" + x for x in d]
        
    value_dict = {}
    labels = []
    imgs = []
    for f1 in f:
        d1 = os.listdir(f1)   
        f2 = [f1 + "/" + x for x in d1]
        label = int(f1.split("_")[0].split("/")[-1])
             
        for img_dir in f2:
            labels.append(label)
            img = Image.open(img_dir)
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_byte = buffer.getvalue()
            imgs.append(img_byte)
            
    value_dict['label'] = labels
    value_dict['image'] = imgs
    
    return (index, value_dict)

def get_img_label(file, index):
    
    img = Image.open(file)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_byte = buffer.getvalue()
        
    return (index, img_byte)

    

def write_lmdb(results, txn):
    for index, value_dict in results:
        # Construct the key-value pair
        key = str(index).encode()
        value = pickle.dumps(value_dict)
        # Store the key-value pair in the database
        txn.put(key, value)
    txn.commit()

    
def main(args):
    time0 = time.time()
    data_path = "../datasets/pretrain/"

    num = args.num
    
    ################################## img ################################
    img_results = Parallel(n_jobs=args.jobs)(
        delayed(get_img_label)(data_path + 'img/' + "{}.png".format(i), i) for i in tqdm(range(num), ncols=120))
    
    env_img = lmdb.open(
            data_path + "img_lmdb",
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
            map_size=1099511627776)
    txn = env_img.begin(write=True)
    write_lmdb(img_results, txn)
    env_img.close()
    
    ################################## Atom ################################
    atom_results = Parallel(n_jobs=args.jobs)(
        delayed(get_atom_bond_motif_label)(data_path + 'Atom/mask/' + str(i), i) for i in tqdm(range(num), ncols=120))
    
    env_atom = lmdb.open(
            data_path + "Atom_lmdb",
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
            map_size=1099511627776)
    txn = env_atom.begin(write=True)
    write_lmdb(atom_results, txn)
    env_atom.close()
    
    ################################## Bond ################################
    bond_results = Parallel(n_jobs=args.jobs)(
        delayed(get_atom_bond_motif_label)(data_path + 'Bond/mask/' + str(i), i) for i in tqdm(range(num), ncols=120))
    
    env_bond = lmdb.open(
            data_path + "Bond_lmdb",
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
            map_size=1099511627776)
    txn = env_bond.begin(write=True)
    write_lmdb(bond_results, txn)
    env_bond.close()
    
    ################################## Motif ################################
    motif_results = Parallel(n_jobs=args.jobs)(
        delayed(get_atom_bond_motif_label)(data_path + 'Motif/mask/' + str(i), i) for i in tqdm(range(num), ncols=120))
    
    env_motif = lmdb.open(
            data_path + "Motif_lmdb",
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
            map_size=1099511627776)
    txn = env_motif.begin(write=True)
    write_lmdb(motif_results, txn)
    env_motif.close()
    
    
    print("Complete! Cost: {} ".format(time.time() - time0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters of Mask Data Process')

    parser.add_argument('--num', default=2000000, type=int, help='data num 2000000')
    parser.add_argument('--jobs', default=10, type=int, help='n_jobs')

    args = parser.parse_args()
    main(args)