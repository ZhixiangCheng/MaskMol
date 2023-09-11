from numpy import load
import pandas as pd
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
from sklearn.model_selection import train_test_split


def train_test_val_idx(dataroot, dataset):

    data = load(dataroot + '{}/raw/split.npz'.format(dataset))

    train_idx = data['train_idx']
    test_idx = data['test_idx']
    val_idx = data['val_idx']

    return train_idx, test_idx, val_idx


def smiles_label(dataroot, dataset, train_idx, test_idx, val_idx):

    data = pd.read_csv(dataroot + '{}/raw/{}.csv'.format(dataset, dataset))

    train_smi = []
    train_label = []

    test_smi = []
    test_label = []

    val_smi = []
    val_label = []

    for i in train_idx:
        train_smi.append(data['smiles'][i])
        train_label.append(data['pIC50'][i])

    for i in test_idx:
        test_smi.append(data['smiles'][i])
        test_label.append(data['pIC50'][i])

    for i in val_idx:
        val_smi.append(data['smiles'][i])
        val_label.append(data['pIC50'][i])

    return train_smi, train_label, test_smi, test_label, val_smi, val_label


def smiles_label_cliffs(dataroot, dataset):

    data = pd.read_csv(dataroot + '{}/{}.csv'.format(dataset, dataset))
    
    train_smi = data[data['split'] == "train"]['smiles'].tolist()
    test_smi =  data[data['split'] == "test"]['smiles'].tolist()

    train_label = data[data['split'] == "train"]['y'].tolist()
    test_label = data[data['split'] == "test"]['y'].tolist()
    
    cliff_mols_train = data[data['split'] == 'train']['cliff_mol'].tolist()
    cliff_mols_test = data[data['split'] == 'test']['cliff_mol'].tolist()
    
    return train_smi, train_label, test_smi, test_label, cliff_mols_train, cliff_mols_test


def smiles_label_SME(dataroot, dataset):

    data = pd.read_csv(dataroot + '{}/{}.csv'.format(dataset, dataset))
    
    train_smi = data[data['group'] == "training"]['smiles'].tolist()
    test_smi =  data[data['group'] == "test"]['smiles'].tolist()
    val_smi =  data[data['group'] == "valid"]['smiles'].tolist()

    train_label = data[data['group'] == "training"]['{}'.format(dataset)].tolist()
    test_label =  data[data['group'] == "test"]['{}'.format(dataset)].tolist()
    val_label =  data[data['group'] == "valid"]['{}'.format(dataset)].tolist()
    
    return train_smi, train_label, test_smi, test_label, val_smi, val_label



class ImageDataset(Dataset):
    def __init__(self, smiles, labels, img_transformer=None, normalize=None):
        self.smiles = smiles
        self.labels = labels
        self.normalize = normalize
        self.img_transformer = img_transformer

    def __len__(self):
        return len(self.smiles)

    def get_image(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224))
        img = Image.frombytes("RGB", img.size, img.tobytes())

        return self.img_transformer(img)

    def __getitem__(self, index):
        smi = self.smiles[index]
        label = self.labels[index]
        img = self.get_image(smi)

        if self.normalize is not None:
            img = self.normalize(img)

        return img, label