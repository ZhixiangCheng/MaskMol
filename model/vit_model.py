import torch.nn as nn
import timm
import torch


class MaskMol(nn.Module):
    def __init__(self, atom_classes=10, bond_classes=4, motif_classes=200, img_size=224, patch_size=16,
                 pretrained=True):
        super(MaskMol, self).__init__()

        self.atom_classes = atom_classes
        self.bond_classes = bond_classes
        self.motif_classes = motif_classes

        self.vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)

        # Remove the classification head to get features only
        self.vit.head = nn.Identity()

        # Calculate the number of patches
        num_patches = (img_size // patch_size) * (img_size // patch_size)

        # Add a classification layer for each patch
        self.atom_patch_classifier = nn.Linear(768, atom_classes)
        self.bond_patch_classifier = nn.Linear(768, bond_classes)
        self.motif_classifier = nn.Linear(768, motif_classes)

    def forward(self, x):
        y = self.vit.forward_features(x)  # Shape: (batch_size, cls(1)+num_patches(196), 768)

        y = y[:, 1:, :]  # (batch_size, 196, 768)

        atom_patch_outputs = self.atom_patch_classifier(y)  # Shape: (batch_size, num_patches, num_classes)

        # Mask Bond task
        bond_patch_outputs = self.bond_patch_classifier(y)

        # Mask Motif task
        motif_ouput = self.vit.forward_head(y)
        motif_outputs = self.motif_classifier(motif_ouput)

        return atom_patch_outputs, bond_patch_outputs, motif_outputs
