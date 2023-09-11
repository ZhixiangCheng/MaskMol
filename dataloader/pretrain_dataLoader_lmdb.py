import os
import random
import torch
import numpy as np
import cv2
from torchvision import transforms
from sklearn.model_selection import train_test_split
from random import sample
from torch.utils.data import Dataset
import lmdb
import pickle
from PIL import Image
from io import BytesIO


class MoleculeDataset_LDMB(Dataset):
    def __init__(self, index, data_path, proportion=0.5, transform=None):
        self.index = index
        self.data_path = data_path
        self.txn_img = None 
        self.txn_atom = None   
        self.txn_bond = None 
        self.txn_motif = None
        self.env = None
        self.transform = transform
        self.proportion = proportion

    def __len__(self):
        return len(self.index)
    
    def _init_db(self):
        
        self.env_img = lmdb.open(self.data_path + "img_lmdb", subdir=False, readonly=True, 
                        lock=False, readahead=False, meminit=False, max_readers=256)
        self.txn_img = self.env_img.begin()
        self.lmdb_env_img = {'txn': self.txn_img, 'keys': list(self.txn_img.cursor().iternext(values=False))}


        self.env_atom = lmdb.open(self.data_path + "Atom_lmdb", subdir=False, readonly=True, 
                            lock=False, readahead=False, meminit=False, max_readers=256)
        self.txn_atom = self.env_atom.begin()
        self.lmdb_env_atom = {'txn': self.txn_atom, 'keys': list(self.txn_atom.cursor().iternext(values=False))}


        self.env_bond = lmdb.open(self.data_path + "Bond_lmdb", subdir=False, readonly=True, 
                            lock=False, readahead=False, meminit=False, max_readers=256)
        self.txn_bond = self.env_bond.begin()
        self.lmdb_env_bond = {'txn': self.txn_bond, 'keys': list(self.txn_bond.cursor().iternext(values=False))}

        self.env_motif = lmdb.open(self.data_path + "Motif_lmdb", subdir=False, readonly=True, 
                            lock=False, readahead=False, meminit=False, max_readers=256)
        self.txn_motif = self.env_motif.begin()
        self.lmdb_env_motif = {'txn': self.txn_motif, 'keys': list(self.txn_motif.cursor().iternext(values=False))}
        

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()
        
        k = self.index[idx]
        proportion = self.proportion
        
        img_datapoint_pickled = self.txn_img.get(str(k).encode())
        img = pickle.loads(img_datapoint_pickled)
        image = self.PIL2Numpy(Image.open(BytesIO(img)))
           
        # mask atom pretext task
        atom_datapoint_pickled = self.txn_atom.get(str(k).encode())
        atom_data = pickle.loads(atom_datapoint_pickled)
        atom_image_mask_np, atom_label_mask, atom_image_mask = self.random_select(atom_data, proportion)
        atom_mask_img = self.get_mask_img(image, atom_image_mask_np)
        atom_patch_labels = self.get_label(self.PIL2Gray(img), atom_image_mask, atom_label_mask)
    
        
        # mask bond pretext task
        img = pickle.loads(img_datapoint_pickled)
        image = self.PIL2Numpy(Image.open(BytesIO(img)))
        
        bond_datapoint_pickled = self.txn_bond.get(str(k).encode())
        bond_data = pickle.loads(bond_datapoint_pickled)
        bond_image_mask_np, bond_label_mask, bond_image_mask = self.random_select(bond_data, proportion)
        bond_mask_img = self.get_mask_img(image, bond_image_mask_np)
        bond_patch_labels = self.get_label(self.PIL2Gray(img), bond_image_mask, bond_label_mask)
        
       
        # mask motif pretext task
        img = pickle.loads(img_datapoint_pickled)
        image = self.PIL2Numpy(Image.open(BytesIO(img)))
        
        motif_datapoint_pickled = self.txn_motif.get(str(k).encode())
        motif_data = pickle.loads(motif_datapoint_pickled)
        motif_mask_img, motif_label = self.get_mask_label_motif(image, motif_data)
        
        if self.transform:
            atom_mask_img = self.transform(Image.fromarray(cv2.cvtColor(atom_mask_img,cv2.COLOR_BGR2RGB)))
            bond_mask_img = self.transform(Image.fromarray(cv2.cvtColor(bond_mask_img,cv2.COLOR_BGR2RGB)))
            motif_mask_img = self.transform(Image.fromarray(cv2.cvtColor(motif_mask_img,cv2.COLOR_BGR2RGB)))
        
        return atom_mask_img, atom_patch_labels, bond_mask_img, bond_patch_labels, motif_mask_img, motif_label
    
    def get_mask_label_motif(self, image, motif_data):
        motif_data = list(zip(motif_data['label'], motif_data['image']))
        data = random.sample(motif_data, 1)
        motif_label = data[0][0]
        img_mask = self.PIL2Numpy(Image.open(BytesIO(data[0][1])))
        image[:, :, :][img_mask[:, :, :] > 0] = 255
        return image, motif_label
    
    def random_select(self, data, proportion):
        # image_no_mask, image_mask, label_no_mask, label_mask = train_test_split(data['image'], data['label'], test_size=proportion, shuffle=True)
        data = list(zip(data['label'], data['image']))
        data = random.sample(data, int(len(data) * proportion))
        image_mask = [x[1] for x in data]
        label_mask = [x[0] for x in data]
        
        image_mask_np = [self.PIL2Numpy(Image.open(BytesIO(x))) for x in image_mask]
        image_mask = [self.PIL2Gray(x) for x in image_mask]
        
        return image_mask_np, label_mask, image_mask
    
    def PIL2Numpy(self, pil_image):     
        # Convert PIL image to numpy array
        numpy_image = np.array(pil_image)
        # Convert RGB to BGR
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        return opencv_image
    
    def PIL2Gray(self, img):
        pil_image = Image.open(BytesIO(img)).convert("L")
        numpy_image = np.array(pil_image)
        return numpy_image
    
    def get_mask_img(self, molecule, image_mask):
        
        img = image_mask[0]  

        for i, m in enumerate(image_mask):

            if len(image_mask) == 1:
                break

            if i == 0:
                img = image_mask[i]
            else:
                img_i = m
                img[:, :, 0] = img[:, :, 0] + img_i[:, :, 0]
                img[:, :, 1] = img[:, :, 1] + img_i[:, :, 1]
                img[:, :, 2] = img[:, :, 2] + img_i[:, :, 2]

        molecule[:, :, :][img[:, :, :] > 0] = 255

        return molecule
    
    def get_label(self, img, img_mask, label_mask):
        molecular_image = img

        masks = {}
        for i, mask in enumerate(img_mask):
            label_type = label_mask[i]
            if label_type not in masks:
                masks[label_type] = []

            masks[label_type].append(mask)

        # Divide the molecular image into 196 patches
        patch_size = 16
        labels = np.empty((14, 14), dtype=object)
        for i in range(14):
            for j in range(14):
                patch = molecular_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]

                # Calculate the number of white pixels belonging to each atom type
                atom_counts = {}
                for atom_type, atom_masks in masks.items():
                    count = 0
                    for mask in atom_masks:
                        intersect = cv2.bitwise_and(patch, mask[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size])
                        #                         print(intersect)
                        count += np.sum(intersect > 0)
                    atom_counts[atom_type] = count

                # Assign the label of the patch based on the atom type with the highest pixel count
                max_count = max(atom_counts.values())
                if max_count > 0:
                    labels[i, j] = [atom_type for atom_type, count in atom_counts.items() if count == max_count][0]
                else:
                    labels[i, j] = -1
        labels = labels.flatten()
        labels = labels.astype(float)
        labels = torch.tensor(labels, dtype=torch.long)

        return labels
