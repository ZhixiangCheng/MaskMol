# -*- coding: UTF-8 -*-
"""
Project -> File: molecule -> train
Author: cer0
Date: 2023-04-14 16:48
Description:

"""
import argparse
import time
import os
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import sys
from tensorboardX import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import math
from torch.utils.data import DataLoader
import lmdb
import pickle
from sklearn.model_selection import train_test_split

from utils.public_utils import setup_device, Logger
from model.train_utils import fix_train_random_seed
from model.vit_model import MaskMol
from dataloader.pretrain_dataLoader_lmdb import MoleculeDataset_LDMB


def env_open(data_path):
    env_img = lmdb.open(data_path + "img_lmdb", subdir=False, readonly=True, 
                        lock=False, readahead=False, meminit=False, max_readers=256)
    txn_img = env_img.begin()
    lmdb_env_img = {'txn': txn_img, 'keys': list(txn_img.cursor().iternext(values=False))}


    env_atom = lmdb.open(data_path + "Atom_lmdb", subdir=False, readonly=True, 
                        lock=False, readahead=False, meminit=False, max_readers=256)
    txn_atom = env_atom.begin()
    lmdb_env_atom = {'txn': txn_atom, 'keys': list(txn_atom.cursor().iternext(values=False))}


    env_bond = lmdb.open(data_path + "Bond_lmdb", subdir=False, readonly=True, 
                        lock=False, readahead=False, meminit=False, max_readers=256)
    txn_bond = env_bond.begin()
    lmdb_env_bond = {'txn': txn_bond, 'keys': list(txn_bond.cursor().iternext(values=False))}

    env_motif = lmdb.open(data_path + "Motif_lmdb", subdir=False, readonly=True, 
                        lock=False, readahead=False, meminit=False, max_readers=256)
    txn_motif = env_motif.begin()
    lmdb_env_motif = {'txn': txn_motif, 'keys': list(txn_motif.cursor().iternext(values=False))}
    
    return lmdb_env_img, lmdb_env_atom, lmdb_env_bond, lmdb_env_motif


def eval(args, dataloader, model):
    total = len(dataloader.dataset)

    returnData = {
        "AtomAcc": 0,
        "BondAcc": 0,
        "MotifAcc": 0,
        "total": 0,
    }

    # evaluation
    with torch.no_grad():
        atom_correct = 0
        bond_correct = 0
        motif_correct = 0
        no_mask_atom_patches_total = 0
        no_mask_bond_pathes_total = 0

        for atom_mask_img, atom_patch_labels, bond_mask_img, bond_patch_labels, motif_mask_img, motif_label in tqdm(
                dataloader, total=len(dataloader), position=0, ncols=120):
            atom_mask_img_var = torch.autograd.Variable(atom_mask_img.cuda())
            bond_mask_img_var = torch.autograd.Variable(bond_mask_img.cuda())
            motif_mask_img_var = torch.autograd.Variable(motif_mask_img.cuda())

            atom_patch_labels_var = torch.autograd.Variable(atom_patch_labels.cuda())
            bond_patch_labels_var = torch.autograd.Variable(bond_patch_labels.cuda())
            motif_label_var = torch.autograd.Variable(motif_label.cuda())

            pre_atom_label, _, _ = model(atom_mask_img_var)
            _, pre_bond_label, _ = model(bond_mask_img_var)
            _, _, pre_motif_label = model(motif_mask_img_var)

            atom_no_mask = (atom_patch_labels_var != -1)
            bond_no_mask = (bond_patch_labels_var != -1)

            # atom_patch_labels_no_mask：Shape: （batch_size * no_mask_num_patches）
            atom_patch_labels_no_mask = atom_patch_labels_var[atom_no_mask]
            bond_patch_labels_no_mask = bond_patch_labels_var[bond_no_mask]

            # pre_atom_label_no_mask：Shape: (batch_size * no_mask_num_patches * num_classes=10)
            pre_atom_label_no_mask = pre_atom_label[atom_no_mask]
            pre_bond_label_no_mask = pre_bond_label[bond_no_mask]

            _, cls_atom_pred = pre_atom_label_no_mask.max(dim=1)
            _, cls_bond_pred = pre_bond_label_no_mask.max(dim=1)
            _, cls_motif_pred = pre_motif_label.max(dim=1)

            atom_correct += torch.sum(cls_atom_pred == atom_patch_labels_no_mask)
            bond_correct += torch.sum(cls_bond_pred == bond_patch_labels_no_mask)
            motif_correct += torch.sum(cls_motif_pred == motif_label_var)

            no_mask_atom_patches_total += atom_patch_labels_no_mask.shape[0]
            no_mask_bond_pathes_total += bond_patch_labels_no_mask.shape[0]

            # atom_correct += torch.sum(torch.all(torch.eq(cls_atom_pred, atom_patch_labels_var), dim=1))
            # bond_correct += torch.sum(torch.all(torch.eq(cls_bond_pred, bond_patch_labels_var), dim=1))

        # atom_acc = float(atom_correct) / (total * 196)  # 196 patches
        # bond_acc = float(bond_correct) / (total * 196)  # 196 patches

        # atom_acc = float(atom_correct) / total  # if every patches are correct, then the img classify correctly
        # bond_acc = float(bond_correct) / total

        atom_acc = float(atom_correct) / no_mask_atom_patches_total
        bond_acc = float(bond_correct) / no_mask_bond_pathes_total
        motif_acc = float(motif_correct) / total  # not need patch

    returnData["AtomAcc"] = atom_acc
    returnData["BondAcc"] = bond_acc
    returnData["MotifAcc"] = motif_acc

    return returnData


def main(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    sys.stdout = Logger(args.log_dir + "pretrain_log_mutiC_40w.log")

    tb_writer = SummaryWriter()

    start_time = time.time()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device, device_ids = setup_device(args.ngpu)

    # fix random seeds
    fix_train_random_seed(args.seed)

    # "No_mask":-1, "C": 0, "N": 1, "O": 2, "S": 3, "F": 4, "Cl": 5, "Br": 6, "I": 7, "P": 8, "Si": 9,
    # "No_mask":-1, "SINGLE": 0, "AROMATIC": 1, "DOUBLE": 2, "TRIPLE": 3,

    mask_atom_classes = 10  # 10 atom types
    mask_bond_classes = 4  # 4 bond types
    mask_motif_classes = 200
    val_size = 0.05
    eval_each_batch = 1000

    ################################## laod data begin ################################

    transform = [
                 transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                 transforms.RandomApply([transforms.RandomAdjustSharpness(2)], p=0.5),
                 transforms.RandomApply([transforms.RandomEqualize()], p=0.5),
                 transforms.RandomGrayscale(p=0.2),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]), ]
    
    #lmdb_env_img, lmdb_env_atom, lmdb_env_bond, lmdb_env_motif = env_open(args.data_path)
    
    train_index, val_index = train_test_split(list(range(args.nums)), test_size=0.05, shuffle=True, random_state=2023)
    
    train_dataset = MoleculeDataset_LDMB(train_index, args.data_path, args.proportion, transforms.Compose(transform))
    val_dataset = MoleculeDataset_LDMB(val_index, args.data_path, args.proportion, transforms.Compose(transform))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                                                   num_workers=args.workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False,
                                                 num_workers=args.val_workers, pin_memory=True)
    print(len(val_dataloader.dataset))

    ################################## laod data End ##################################

    ################################## laod modle begin ###############################

    model = MaskMol(atom_classes=mask_atom_classes, bond_classes=mask_bond_classes, motif_classes=mask_motif_classes)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loading completed")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if len(device_ids) > 1:
        print("starting multi-gpu.")
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    model = model.cuda()

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10 ** args.wd,
    )
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)

    # define loss function
    criterion = torch.nn.CrossEntropyLoss().cuda()

    ################################## laod modle end #################################

    ################################## start pretrain #################################

    for epoch in range(args.start_epoch, args.epochs):

        # switch to train mode
        model.train()

        AvgMaskAtomLoss = 0
        AvgMaskBondLoss = 0
        AvgMaskMotifLoss = 0
        AvgTotalLoss = 0

        with tqdm(total=len(train_dataloader), position=0, ncols=120) as t:
            for i, (atom_mask_img, atom_patch_labels, bond_mask_img, bond_patch_labels, motif_mask_img,
                    motif_label) in enumerate(train_dataloader):
                
                atom_mask_img_var = torch.autograd.Variable(atom_mask_img.cuda())
                bond_mask_img_var = torch.autograd.Variable(bond_mask_img.cuda())
                motif_mask_img_var = torch.autograd.Variable(motif_mask_img.cuda())

                # atom_patch_labels_var: Shape: （batch_size * num_patches）
                atom_patch_labels_var = torch.autograd.Variable(atom_patch_labels.cuda())
                bond_patch_labels_var = torch.autograd.Variable(bond_patch_labels.cuda())
                motif_label_var = torch.autograd.Variable(motif_label.cuda())

                # pre_atom_label：Shape: (batch_size * num_patches=196 * num_classes=11)
                pre_atom_label, _, _ = model(atom_mask_img_var)
                _, pre_bond_label, _ = model(bond_mask_img_var)
                _, _, pre_motif_label = model(motif_mask_img_var)

                atom_no_mask = (atom_patch_labels_var != -1)
                bond_no_mask = (bond_patch_labels_var != -1)

                # atom_patch_labels_no_mask：Shape: （batch_size * no_mask_num_patches）
                atom_patch_labels_no_mask = atom_patch_labels_var[atom_no_mask]
                bond_patch_labels_no_mask = bond_patch_labels_var[bond_no_mask]

                # pre_atom_label_no_mask：Shape: (batch_size * no_mask_num_patches * num_classes=10)
                pre_atom_label_no_mask = pre_atom_label[atom_no_mask]
                pre_bond_label_no_mask = pre_bond_label[bond_no_mask]

                # only use the mask patches to calculate atom_loss/bond_loss
                atom_loss = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
                if args.Atom_lambda != 0:
                    atom_loss = criterion(pre_atom_label_no_mask, atom_patch_labels_no_mask)

                bond_loss = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
                if args.Bond_lambda != 0:
                    bond_loss = criterion(pre_bond_label_no_mask, bond_patch_labels_no_mask)

                motif_loss = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
                if args.Motif_lambda != 0:
                    # pre_motif_label: # Shape: (batch_size * num_classes=200)
                    motif_loss = criterion(pre_motif_label, motif_label_var)

                # calculating all loss to backward
                loss = atom_loss * args.Atom_lambda + bond_loss * args.Bond_lambda + motif_loss * args.Motif_lambda

                # calculating average loss
                AvgMaskAtomLoss += atom_loss.item() / len(train_dataloader)
                AvgMaskBondLoss += bond_loss.item() / len(train_dataloader)
                AvgMaskMotifLoss += motif_loss.item() / len(train_dataloader)
                AvgTotalLoss += loss.item() / len(train_dataloader)

                # compute gradient and do SGD step
                if loss.item() != 0:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                if args.verbose and (i % eval_each_batch) == 0:
                    print('Epoch: [{}][{}/{}]\t'
                          'TotalLoss：{:.3f}\t'
                          'AtomLoss：{:.3f}\t'
                          'BondLoss：{:.3f}\t'
                          'MotifLoss：{:.3f}\t'
                          .format(epoch + 1, i, len(train_dataloader), loss.item(),
                                   atom_loss.item(), bond_loss.item(), motif_loss.item()))

                t.set_postfix(TotalLoss=loss.item(), AtomLoss=atom_loss.item(), BondLoss=bond_loss.item(),
                              MotifLoss=motif_loss.item())
                t.update(1)

        scheduler.step()

        ################################## start evaluation #################################
        model.eval()
        evaluationData = eval(args, val_dataloader, model)

        # save model
        saveRoot = os.path.join(args.ckpt_dir, 'checkpoints')
        if not os.path.exists(saveRoot):
            os.makedirs(saveRoot)
        if epoch % args.checkpoints == 0:
            saveFile = os.path.join(saveRoot, 'MaskMol_{}.pth.tar'.format(epoch + 1))
            if args.verbose:
                print('Save checkpoint at: {}'.format(saveFile))

            if isinstance(model, torch.nn.DataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            torch.save({
                'state_dict': model_state_dict,
            }, saveFile)

        print('Epoch: [{}][train]\t'
              'TotalLoss：{:.3f}\t'
              'AtomLoss：{:.3f}\t'
              'BondLoss：{:.3f}\t'
              'MotifLoss：{:.3f}\t'
              .format(epoch + 1, AvgTotalLoss, AvgMaskAtomLoss, AvgMaskBondLoss, AvgMaskMotifLoss))

        print('Epoch: [{}][val]\t'
              'AtomAcc：{:.3f}\t'
              'BondAcc：{:.3f}\t'
              'MotifAcc：{:.3f}\t\n'
              .format(epoch + 1, evaluationData['AtomAcc'], evaluationData['BondAcc'], evaluationData['MotifAcc']))

        tags = ["TotalLoss", "AtomLoss", "AtomAcc", "BondLoss", "BondAcc", "MotifLoss", "MotifAcc"]

        tb_writer.add_scalar(tags[0], AvgTotalLoss, epoch + 1)
        tb_writer.add_scalar(tags[1], AvgMaskAtomLoss, epoch + 1)
        tb_writer.add_scalar(tags[2], evaluationData['AtomAcc'], epoch + 1)
        tb_writer.add_scalar(tags[3], AvgMaskBondLoss, epoch + 1)
        tb_writer.add_scalar(tags[4], evaluationData['BondAcc'], epoch + 1)
        tb_writer.add_scalar(tags[5], AvgMaskMotifLoss, epoch + 1)
        tb_writer.add_scalar(tags[6], evaluationData['MotifAcc'], epoch + 1)

    print("used time: {}".format(time.time() - start_time))

    # sys.stdout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters of pretraining MMViT')

    parser.add_argument('--lr', default=0.01, type=float, help='learning rate (default: 0.01)')
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--nums', type=int, default=200000)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--workers', default=30, type=int, help='number of data loading workers (default: 2)')
    parser.add_argument('--val_workers', default=25, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--epochs', type=int, default=50, help='number of total epochs to run (default: 151)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=256, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--checkpoints', type=int, default=1,
                        help='how many iterations between two checkpoints (default: 1)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint 85798656 (default: None) ./ckpts/pretrain/checkpoints/MMViT_1.pth.tar')
    parser.add_argument('--seed', type=int, default=2023, help='random seed (default: 2023)')
    parser.add_argument('--data_path', type=str, default="./datasets/pretrain/", help='data root')
    parser.add_argument('--data_dir', type=str, default="./datasets/pretrain/img/", help='data root')
    parser.add_argument('--mask_atom_dir', type=str, default="./datasets/pretrain/Atom/mask/", help='mask atom dir')
    parser.add_argument('--mask_bond_dir', type=str, default="./datasets/pretrain/Bond/mask/", help='mask bond dir')
    parser.add_argument('--mask_motif_dir', type=str, default="./datasets/pretrain/Motif/mask/", help='mask motif dir')
    parser.add_argument('--log_dir', type=str, default="./logs/", help='log dir')
    parser.add_argument('--proportion', type=float, default=0.1, help='mask atom or bond proportion')
    parser.add_argument('--ckpt_dir', default='./ckpts/pretrain', help='path to checkpoint')
    parser.add_argument('--verbose', action='store_true', help='')
    parser.add_argument('--ngpu', type=int, default=8, help='number of GPUs to use')
    parser.add_argument('--gpu', type=str, default="0", help='GPUs of CUDA_VISIBLE_DEVICES')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--Atom_lambda', type=float, default=1,
                        help='start MAP(Mask Atom Prediction) task, 1 means start, 0 means not start')
    parser.add_argument('--Bond_lambda', type=float, default=1,
                        help='start MBP(Mask Bond Prediction) task, 1 means start, 0 means not start')
    parser.add_argument('--Motif_lambda', type=float, default=1,
                        help='start MMP(Mask Motif Prediction) task, 1 means start, 0 means not start')
    args = parser.parse_args()

    main(args)
