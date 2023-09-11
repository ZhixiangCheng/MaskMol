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

# [*] Packages required to import distributed data parallelism
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# [*] Initialize the distributed process group and distributed device
def setup_DDP_mp(init_method, local_rank, rank, world_size, backend="nccl", verbose=False):
    # If the OS is Windows or macOS, use gloo instead of nccl
    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
    # set distributed device
    device = torch.device("cuda:{}".format(local_rank))
    if verbose:
        print("Using device: {}".format(device))
        print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
    return device


def print_only_rank0(log):
    if dist.get_rank() == 0:
        print(log)

        
def parse_args():
    parser = argparse.ArgumentParser(description='parameters of pretraining MMViT')

    parser.add_argument('--lr', default=0.01, type=float, help='learning rate (default: 0.01)')
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--nums', type=int, default=2000000)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--workers', default=15, type=int, help='number of data loading workers (default: 2)')
    parser.add_argument('--val_workers', default=15, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--epochs', type=int, default=50, help='number of total epochs to run (default: 50)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--checkpoints', type=int, default=1,
                        help='how many iterations between two checkpoints (default: 1)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--seed', type=int, default=2023, help='random seed (default: 2023)')
    parser.add_argument('--data_path', type=str, default=".datasets/pretrain/", help='data root')
    parser.add_argument('--log_dir', type=str, default="./logs/", help='log dir')
    parser.add_argument('--proportion', type=float, default=0.5, help='mask atom or bond proportion')
    parser.add_argument('--ckpt_dir', default='./ckpts/', help='path to checkpoint')
    parser.add_argument('--verbose', action='store_true', help='')
    parser.add_argument('--ngpu', type=int, default=4, help='number of GPUs to use')
    parser.add_argument('--gpu', type=str, default="0", help='GPUs of CUDA_VISIBLE_DEVICES')
    parser.add_argument('--Atom_lambda', type=float, default=1,
                        help='start MAP(Mask Atom Prediction) task, 1 means start, 0 means not start')
    parser.add_argument('--Bond_lambda', type=float, default=1,
                        help='start MBP(Mask Bond Prediction) task, 1 means start, 0 means not start')
    parser.add_argument('--Motif_lambda', type=float, default=1,
                        help='start MMP(Mask Motif Prediction) task, 1 means start, 0 means not start')
    
    parser.add_argument("--nodes", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--ngpus_per_node", default=4, type=int, help="number of GPUs per node for distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:12355", type=str, help="url used to set up distributed training")
    parser.add_argument("--node_rank", default=0, type=int, help="node rank for distributed training")
    return parser.parse_args()


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
        

def eval(args, dataloader, model, device):
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
        motif_total = 0

        for atom_mask_img, atom_patch_labels, bond_mask_img, bond_patch_labels, motif_mask_img, motif_label in tqdm(
                dataloader, total=len(dataloader), position=0, ncols=120):
            atom_mask_img_var = torch.autograd.Variable(atom_mask_img.to(device))
            bond_mask_img_var = torch.autograd.Variable(bond_mask_img.to(device))
            motif_mask_img_var = torch.autograd.Variable(motif_mask_img.to(device))

            atom_patch_labels_var = torch.autograd.Variable(atom_patch_labels.to(device))
            bond_patch_labels_var = torch.autograd.Variable(bond_patch_labels.to(device))
            motif_label_var = torch.autograd.Variable(motif_label.to(device))

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
            motif_total += motif_label_var.shape[0]

        atom_acc = float(atom_correct) / no_mask_atom_patches_total
        bond_acc = float(bond_correct) / no_mask_bond_pathes_total
        motif_acc = float(motif_correct) / motif_total  # not need patch

    returnData["AtomAcc"] = atom_acc
    returnData["BondAcc"] = bond_acc
    returnData["MotifAcc"] = motif_acc

    return returnData


def main(local_rank, ngpus_per_node, args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    sys.stdout = Logger(args.log_dir + "pretrain_log_{}.log".format(args.proportion))

    tb_writer = SummaryWriter()

    start_time = time.time()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    args.local_rank = local_rank
    args.rank = args.node_rank * ngpus_per_node + local_rank
    
    device = setup_DDP_mp(init_method=args.dist_url, local_rank=args.local_rank, rank=args.rank,
                          world_size=args.world_size, verbose=True)

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

    # lmdb_env_img, lmdb_env_atom, lmdb_env_bond, lmdb_env_motif = env_open(args.data_path)
    
    train_index, val_index = train_test_split(list(range(args.nums)), test_size=0.05, shuffle=True, random_state=2023)
    
    train_dataset = MoleculeDataset_LDMB(train_index, args.data_path, args.proportion, transforms.Compose(transform))
    val_dataset = MoleculeDataset_LDMB(val_index, args.data_path, args.proportion, transforms.Compose(transform))
    
    batch_size = args.batch // args.world_size  # [*] // world_size
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)  # [*]
    test_sampler = DistributedSampler(val_dataset, shuffle=False)  # [*]
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=args.workers, pin_memory=True)  # [*] sampler=...
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=args.val_workers, pin_memory=True)  # [*] sampler=...


    ################################## laod data End ##################################

    ################################## laod modle begin ###############################

    model = MaskMol(atom_classes=mask_atom_classes, bond_classes=mask_bond_classes, motif_classes=mask_motif_classes)
    

    if args.resume:
        if os.path.isfile(args.resume):
            print_only_rank0("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.local_rank))
            # model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
            print_only_rank0("=> loading completed")
        else:
            print_only_rank0("=> no checkpoint found at '{}'".format(args.resume))

    model = model.to(device) 
    
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)  # [*] DDP(...)

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10 ** args.wd,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)

    # define loss function
    criterion = torch.nn.CrossEntropyLoss().cuda()

    ################################## laod modle end #################################

    ################################## start pretrain #################################

    for epoch in range(args.start_epoch, args.epochs):
        
        # [*] set sampler
        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)

        # switch to train mode
        model.train()

        AvgMaskAtomLoss = 0
        AvgMaskBondLoss = 0
        AvgMaskMotifLoss = 0
        AvgTotalLoss = 0

        with tqdm(total=len(train_dataloader), position=0, ncols=120) as t:
            for i, (atom_mask_img, atom_patch_labels, bond_mask_img, bond_patch_labels, motif_mask_img,
                    motif_label) in enumerate(train_dataloader):
                
                atom_mask_img_var = torch.autograd.Variable(atom_mask_img.to(device) )
                bond_mask_img_var = torch.autograd.Variable(bond_mask_img.to(device) )
                motif_mask_img_var = torch.autograd.Variable(motif_mask_img.to(device) )

                # atom_patch_labels_var: Shape: （batch_size * num_patches）
                atom_patch_labels_var = torch.autograd.Variable(atom_patch_labels.to(device) )
                bond_patch_labels_var = torch.autograd.Variable(bond_patch_labels.to(device) )
                motif_label_var = torch.autograd.Variable(motif_label.to(device) )

                # pre_atom_label：Shape: (batch_size * num_patches=196 * num_classes=11)

                # only use the mask patches to calculate atom_loss/bond_loss
                atom_loss = torch.autograd.Variable(torch.Tensor([0.0])).to(device) 
                if args.Atom_lambda != 0:
                    pre_atom_label, _, _ = model(atom_mask_img_var)
                    atom_no_mask = (atom_patch_labels_var != -1)
                    atom_patch_labels_no_mask = atom_patch_labels_var[atom_no_mask]
                    pre_atom_label_no_mask = pre_atom_label[atom_no_mask]
                    atom_loss = criterion(pre_atom_label_no_mask, atom_patch_labels_no_mask)

                bond_loss = torch.autograd.Variable(torch.Tensor([0.0])).to(device) 
                if args.Bond_lambda != 0:
                    _, pre_bond_label, _ = model(bond_mask_img_var)
                    bond_no_mask = (bond_patch_labels_var != -1)
                    bond_patch_labels_no_mask = bond_patch_labels_var[bond_no_mask]
                    pre_bond_label_no_mask = pre_bond_label[bond_no_mask]
                    bond_loss = criterion(pre_bond_label_no_mask, bond_patch_labels_no_mask)

                motif_loss = torch.autograd.Variable(torch.Tensor([0.0])).to(device) 
                if args.Motif_lambda != 0:
                    # pre_motif_label: # Shape: (batch_size * num_classes=200)
                    _, _, pre_motif_label = model(motif_mask_img_var)
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

                if dist.get_rank() == 0 and args.verbose and (i % eval_each_batch) == 0:
                    print('Epoch: [{}][{}/{}]\t'
                          'TotalLoss：{:.3f}\t'
                          'AtomLoss：{:.3f}\t'
                          'BondLoss：{:.3f}\t'
                          'MotifLoss：{:.3f}\t'
                          .format(epoch + 1, i, len(train_dataloader), loss.item(),
                                   atom_loss.item(), bond_loss.item(), motif_loss.item()))
                    
                if dist.get_rank() == 0:
                    t.set_postfix(TotalLoss=loss.item(), AtomLoss=atom_loss.item(), BondLoss=bond_loss.item(),
                                  MotifLoss=motif_loss.item())
                    t.update(1)

        scheduler.step()

        ################################## start evaluation #################################
        model.eval()
        evaluationData = eval(args, val_dataloader, model, device)
        
        # [*] save model on rank 0
        if dist.get_rank() == 0:
            # save model
            saveRoot = os.path.join(args.ckpt_dir, 'checkpoints_{}'.format(args.proportion))
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

    print_only_rank0("used time: {}".format(time.time() - start_time))

    # sys.stdout.close()


if __name__ == '__main__':
    # [*] initialize some arguments
    args = parse_args()
    args.world_size = args.ngpus_per_node * args.nodes
    
    # [*] run with torch.multiprocessing
    mp.spawn(main, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
