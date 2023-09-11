import argparse
import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import timm
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout

# device
def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MaskMol(nn.Module):
    def __init__(self, pretrained=True):
        super(MaskMol, self).__init__()

        self.vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
        # Remove the classification head to get features only
        self.vit.head = nn.Identity()
        
        self.atom_patch_classifier = nn.Linear(768, 10)
        self.bond_patch_classifier = nn.Linear(768, 4)
        self.motif_classifier = nn.Linear(768, 200)
        
        self.regressor = nn.Linear(768, 1)

    def forward(self, x):
        x = self.vit(x)
        x = self.regressor(x)
        return x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='1.png',
                        help='Input image path')
    parser.add_argument('--molecule_path', type=str, default='2.png',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: None) ')
    parser.add_argument('--gpu', default='1', type=str, help='index of GPU to use')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    model = MaskMol()
    # model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    
#     resume = "../ckpts/pretrain/" + args.resume + ".pth.tar"
    resume = "../ckpts/finetuning/{}.pth".format(args.resume)

    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            # del_keys = ['atom_patch_classifier.weight', 'atom_patch_classifier.bias', 'bond_patch_classifier.weight', 'bond_patch_classifier.bias'] 
            # for k in del_keys:
            #     del checkpoint['state_dict'][k]
            model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()}, strict=False)
        else:
            print("=> no checkpoint found at '{}'".format(resume))
    
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(args.image_path)
    img = img.resize((224, 224))
    
#     molecule = Image.open(args.molecule_path)
#     molecule = molecule.resize((224, 224))
    
    input_tensor = transform(img).unsqueeze(0)
    if args.use_cuda:
        input_tensor = input_tensor.cuda()

    if args.category_index is None:
        print("Doing Attention Rollout")
        attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, 
            discard_ratio=args.discard_ratio)
        mask = attention_rollout(input_tensor)
        name = "./output/attention_rollout_{:.3f}_{}_{}.png".format(args.discard_ratio, args.head_fusion, args.resume)
    else:
        print("Doing Gradient Attention Rollout")
        grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
        mask = grad_rollout(input_tensor, args.category_index)
        name = "./output/single_atom_grad_rollout_{}_{:.3f}_{}_{}.png".format(args.category_index,
            args.discard_ratio, args.head_fusion, args.resume)

    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    
    b, g, r = cv2.split(mask)
    mask = cv2.merge((r, g, b))

    plt.imshow(mask)
    plt.axis('off')
    plt.imsave(name, mask)
    plt.show()
