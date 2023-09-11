import os
import random
import numpy as np
import torch
from sklearn import metrics
import logging
import torch.nn as nn
import timm
import torch
from typing import List, Union


class MaskMol(nn.Module):
    def __init__(self, pretrained=True):
        super(MaskMol, self).__init__()

        self.vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
        # Remove the classification head to get features only
        self.vit.head = nn.Identity()
        
        # self.atom_patch_classifier = nn.Linear(768, 10)
        # self.bond_patch_classifier = nn.Linear(768, 4)
        # self.motif_classifier = nn.Linear(768, 200)

        self.regressor = nn.Linear(768, 1)

    def forward(self, x):
        x = self.vit(x)
#         x = torch.relu(x)
        x = self.regressor(x)

        return x


def fix_train_random_seed(seed=2021):
    # fix random seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def save_finetune_ckpt(resume, model, optimizer, loss, epoch, save_path, filename_pre, lr_scheduler=None,
                       result_dict=None,
                       logger=None):
    log = logger if logger is not None else logging
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    lr_scheduler = None if lr_scheduler is None else lr_scheduler.state_dict()
    state = {
        'epoch': epoch,
        'model_state_dict': model_cpu,
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler,
        'loss': loss,
        'result_dict': result_dict
    }
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        log.info("Directory {} is created.".format(save_path))

    filename = '{}/{}_{}.pth'.format(save_path, resume, filename_pre)
    torch.save(state, filename)
    log.info('model has been saved as {}'.format(filename))


def metric_reg(y_true, y_pred):
    '''
    for regression evaluation on single task
    :param y_true: 1-D, e.g. [1.1, 0.2, 1.5, 3.2]
    :param y_pred: 1-D, e.g. [-0.2, 1.1, 1.2, 3.1]
    :return:
    '''
    assert len(y_true) == len(y_pred)
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()

    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }


def calc_rmse(true, pred):
    """ Calculates the Root Mean Square Error

    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)

    Returns: (float) rmse
    """
    # Convert to 1-D numpy array if it's not
    if type(pred) is not np.array:
        pred = np.array(pred)
    if type(true) is not np.array:
        true = np.array(true)

    return np.sqrt(np.mean(np.square(true - pred)))


def calc_cliff_rmse(y_test_pred: Union[List[float], np.array], y_test: Union[List[float], np.array],
                    cliff_mols_test: List[int] = None, smiles_test: List[str] = None,
                    y_train: Union[List[float], np.array] = None, smiles_train: List[str] = None, **kwargs):
    """ Calculate the RMSE of activity cliff compounds

    :param y_test_pred: (lst/array) predicted test values
    :param y_test: (lst/array) true test values
    :param cliff_mols_test: (lst) binary list denoting if a molecule is an activity cliff compound
    :param smiles_test: (lst) list of SMILES strings of the test molecules
    :param y_train: (lst/array) train labels
    :param smiles_train: (lst) list of SMILES strings of the train molecules
    :param kwargs: arguments for ActivityCliffs()
    :return: float RMSE on activity cliff compounds
    """

    # Check if we can compute activity cliffs when pre-computed ones are not provided.
    if cliff_mols_test is None:
        if smiles_test is None or y_train is None or smiles_train is None:
            raise ValueError('if cliff_mols_test is None, smiles_test, y_train, and smiles_train should be provided '
                             'to compute activity cliffs')

    # Convert to numpy array if it is none
    y_test_pred = np.array(y_test_pred) if type(y_test_pred) is not np.array else y_test_pred
    y_test = np.array(y_test) if type(y_test) is not np.array else y_test

    # Get the index of the activity cliff molecules
    cliff_test_idx = [i for i, cliff in enumerate(cliff_mols_test) if cliff == 1]

    # Filter out only the predicted and true values of the activity cliff molecules
    y_pred_cliff_mols = y_test_pred[cliff_test_idx]
    y_test_cliff_mols = y_test[cliff_test_idx]

    return calc_rmse(y_pred_cliff_mols, y_test_cliff_mols)
