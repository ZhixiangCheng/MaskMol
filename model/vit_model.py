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
        # self.vit = timm.create_model("vit_base_patch16_224", num_classes=0)

        # # Change the number of attention heads in each layer to 6
        # num_heads = 6
        # for layer in self.vit .blocks:
        #     layer.attn = timm.models.vision_transformer.Attention(
        #         dim=layer.attn.dim,
        #         num_heads=num_heads,
        #         qkv_bias=layer.attn.qkv_bias,
        #         qk_scale=layer.attn.qk_scale,
        #         attn_drop=layer.attn.attn_drop,
        #         proj_drop=layer.attn.proj_drop,
        #     )

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
        # atom_patch_outputs = patch_outputs.permute(0, 2, 1)    # Shape: (batch_size, num_classes, num_patches)

        # Mask Bond task
        bond_patch_outputs = self.bond_patch_classifier(y)
        # bond_patch_outputs = patch_outputs.permute(0, 2, 1)

        # Mask Motif task
        motif_ouput = self.vit.forward_head(y)
        motif_outputs = self.motif_classifier(motif_ouput)

        return atom_patch_outputs, bond_patch_outputs, motif_outputs
#         return bond_patch_outputs, motif_outputs
