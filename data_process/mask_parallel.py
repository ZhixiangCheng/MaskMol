# -*- coding: UTF-8 -*-
"""
Project -> File: molecule -> mask_parallel
Author: cer0
Date: 2023-04-14 14:13
Description:

"""

import argparse
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
import io
from tqdm import *
import time
import concurrent.futures
from joblib import Parallel, delayed

ATOM_TYPE = {
    "C": 0,
    "N": 1,
    "O": 2,
    "S": 3,
    "F": 4,
    "Cl": 5,
    "Br": 6,
    "I": 7,
    "P": 8,
    "Si": 9,
}

BOND_TYPE = {
    "SINGLE": 0,
    "AROMATIC": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
}


def mask_with_atom_and_bound_index(mol, w, h, atom_list, atom_colour_list, bond_list, bound_colour_list,
                                   radius_list, path="", save_svg=False):
    if not save_svg:
        d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
        d2d.drawOptions().useBWAtomPalette()
        d2d.drawOptions().highlightBondWidthMultiplier = 20
        rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol, highlightAtoms=atom_list,
                                           highlightAtomColors=atom_colour_list,
                                           highlightBonds=bond_list,
                                           highlightBondColors=bound_colour_list,
                                           highlightAtomRadii=radius_list)
        image_data = d2d.GetDrawingText()

        # Convert the byte array to a PIL Image object
        image = Image.open(io.BytesIO(image_data))

        # Convert the PIL Image object to a NumPy array
        image_np = np.array(image)

        return image_np


    else:
        d2d = rdMolDraw2D.MolDraw2DSVG(w, h)
        rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol, highlightAtoms=atom_list,
                                           highlightAtomColors=atom_colour_list,
                                           highlightBonds=bond_list,
                                           highlightBondColors=bound_colour_list,
                                           highlightAtomRadii=radius_list)
        d2d.FinishDrawing()
        svg = d2d.GetDrawingText()
        with open(path, 'w') as f:
            f.write(svg)


def get_color_dict(mol, atom, radius=0.4, color=(0, 1, 0)):
    radius_dict = {}
    atom_colour_dict = {}
    atom_index = []

    # radiux
    radius_dict[atom[1]] = radius

    # color
    atom_colour_dict[atom[1]] = color

    # atom/bond index
    atom_index.append(atom[1])

    # atom/bond type
    atom_type = atom[0]

    return atom_colour_dict, radius_dict, atom_index, atom_type


def get_Atom_Bond_index(mol, data):
    atom_index = {}
    bond_index = {}
    idx = []

    for index in range(len(data['smiles'])):

        patt = Chem.MolFromSmiles(data['smiles'][index])

        # get atom index
        hit_ats = mol.GetSubstructMatches(patt)

        if len(hit_ats) > 0 and len(hit_ats[0]) >= 2:

            idx.append(index)

            bond_lists = []

            for i, hit_at in enumerate(hit_ats):
                hit_at = list(hit_at)
                bond_list = []
                for bond in patt.GetBonds():
                    a1 = hit_at[bond.GetBeginAtomIdx()]
                    a2 = hit_at[bond.GetEndAtomIdx()]
                    bond_list.append(mol.GetBondBetweenAtoms(a1, a2).GetIdx())
                bond_lists.append(bond_list)

            atom_index[index] = hit_ats
            bond_index[index] = bond_lists

    return idx, atom_index, bond_index


def mask_atom(img):
    ball_color = 'green'
    color_dist = {
        'green': {'Lower': np.array([30, 66, 35]), 'Upper': np.array([85, 255, 255])},
    }

    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(img1, (7, 7), 0)
    inRange_hsv = cv2.inRange(blurred, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])

    return inRange_hsv


def mask_bond(img):
    ball_color = 'green'
    color_dist = {
        'green': {'Lower': np.array([30, 120, 40]), 'Upper': np.array([60, 255, 255])},
    }

    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(img1, (7, 7), 0)

    inRange_hsv = cv2.inRange(blurred, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))  # 定义结构元素的形状和大小
    dilated = cv2.dilate(inRange_hsv, kernel)

    return inRange_hsv


def mask_motif(img):
    ball_color = 'green'
    color_dist = {
        'green': {'Lower': np.array([30, 66, 35]), 'Upper': np.array([90, 255, 255])},
    }

    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(img1, (7, 7), 0)

    inRange_hsv = cv2.inRange(blurred, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义结构元素的形状和大小
    dilated = cv2.dilate(inRange_hsv, kernel)

    return inRange_hsv


def Mask_Atom(data_path, index, mol):
    Atom = [(atom.GetSymbol(), atom.GetIdx()) for atom in mol.GetAtoms()]

    for a in Atom:

        if a[0] not in ATOM_TYPE:
            continue

        atom_colour_dict, radius_dict, atom_index, atom_type = get_color_dict(mol, a, radius=0.4, color=(0, 1, 0))

        highlight_img = mask_with_atom_and_bound_index(mol, w=224, h=224,
                                                       atom_list=atom_index,
                                                       atom_colour_list=atom_colour_dict,
                                                       bond_list=None,
                                                       bound_colour_list=None,
                                                       radius_list=radius_dict,
                                                       save_svg=False)

        dilated_atom = mask_atom(highlight_img)

        # save mask image

        dirs = data_path + '/Atom/mask/{}/{}_{}'.format(index, ATOM_TYPE[atom_type], atom_type)

        if not os.path.exists(dirs):
            os.makedirs(dirs)

        path = data_path + '/Atom/mask/{}/{}_{}/{}.png'.format(index, ATOM_TYPE[atom_type], atom_type,
                                                               atom_index[0])

        cv2.imwrite(path, dilated_atom)


def Mask_Bond(data_path, index, mol):
    Bond = [(str(bond.GetBondType()).split(".")[-1], bond.GetIdx()) for bond in mol.GetBonds()]

    for b in Bond:

        bond_colour_dict, radius_dict, bond_index, bond_type = get_color_dict(mol, b, radius=0.4, color=(0, 1, 0))

        highlight_img = mask_with_atom_and_bound_index(mol, w=224, h=224,
                                                       atom_list=None,
                                                       atom_colour_list=None,
                                                       bond_list=bond_index,
                                                       bound_colour_list=bond_colour_dict,
                                                       radius_list=radius_dict,
                                                       save_svg=False)

        dilated_bond = mask_bond(highlight_img)

        # save mask image

        dirs = data_path + '/Bond/mask/{}/{}_{}'.format(index, BOND_TYPE[bond_type], bond_type)

        if not os.path.exists(dirs):
            os.makedirs(dirs)

        path = data_path + '/Bond/mask/{}/{}_{}/{}.png'.format(index, BOND_TYPE[bond_type], bond_type,
                                                               bond_index[0])

        cv2.imwrite(path, dilated_bond)


def Mask_Motif(data_path, index, mol, motif):
    idx, atom, bond = get_Atom_Bond_index(mol, motif)

    for idex in idx:
        atoms = atom[idex]
        bonds = bond[idex]

        for i in range(len(atoms)):

            # color, radius
            radius_dict = {}
            atom_colour_dict = {}
            bond_colour_dict = {}

            for x in atoms[i]:
                radius_dict[x] = 0.4
                atom_colour_dict[x] = (0, 1, 0)

            for x in bonds[i]:
                bond_colour_dict[x] = (0, 1, 0)

            highlight_img = mask_with_atom_and_bound_index(mol, w=224, h=224,
                                                           atom_list=atoms[i],
                                                           atom_colour_list=atom_colour_dict,
                                                           bond_list=bonds[i],
                                                           bound_colour_list=bond_colour_dict,
                                                           radius_list=radius_dict, save_svg=False)

            dilated_motif = mask_bond(highlight_img)

            # save mask image

            dirs = data_path + '/Motif/mask/{}/{}_'.format(index, idex)

            if not os.path.exists(dirs):
                os.makedirs(dirs)

            path = data_path + '/Motif/mask/{}/{}_/{}.png'.format(index, idex, i)

            cv2.imwrite(path, dilated_motif)


def get_img(data_path, index, mol):
    # save molecule image
    dirs = data_path + "/img/"

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    img_path = dirs + "{}.png".format(index)
    Draw.MolToFile(mol, img_path, size=(224, 224))


def Mask_Atom_Bond_Motif(data_path, smiles, motif, index):
    mol = Chem.MolFromSmiles(smiles)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(get_img, data_path, index, mol)

        executor.submit(Mask_Atom, data_path, index, mol)

        executor.submit(Mask_Bond, data_path, index, mol)

        executor.submit(Mask_Motif, data_path, index, mol, motif)


def main(args):
    time0 = time.time()

    data_path = "../datasets/pretrain"

    data = pd.read_csv(data_path + "/{}.csv".format(args.data_dir))
    motif = pd.read_csv(data_path + "/motif.csv")

    smiles_list = list(data['smiles'])[:args.num]

    # smiles_list = [smiles_list[x] for x in idx]

    Parallel(n_jobs=args.jobs)(
        delayed(Mask_Atom_Bond_Motif)(data_path, smiles_list[i], motif, i) for i in tqdm(range(len(smiles_list)), ncols=120))

    #     Mask_Atom_Bond_Motif(data_path, smiles_list, motif)

    print("Complete! Cost: {} ".format(time.time() - time0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters of Mask Data Process')

    parser.add_argument('--data_dir', default="pretrain", type=str, help='file name')
    parser.add_argument('--num', default=2000000, type=int, help='data num 2000000')
    parser.add_argument('--jobs', default=10, type=int, help='n_jobs')

    args = parser.parse_args()
    main(args)

