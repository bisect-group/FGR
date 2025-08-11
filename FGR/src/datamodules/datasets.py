from typing import List

import numpy as np
import torch
from rdkit import RDLogger
from rdkit.Chem.rdmolfiles import MolFromSmarts
from tokenizers import Tokenizer

from FGR.src.datamodules.components.dataset import BaseDataset

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class FGRDataset(BaseDataset):
    def __init__(
        self,
        smiles: List[str],
        labels: np.ndarray,
        fgroups_list: List[str],
        tokenizer: Tokenizer,
        descriptors: bool,
        method: str,
    ) -> None:
        """Dataset for training and testing.

        Args:
            smiles (List[str]): List of SMILES
            labels (np.ndarray): Labels
            fgroups_list (List[str]): List of functional groups
            tokenizer (Tokenizer): Pretrained tokenizer
            descriptors (bool): Whether to use descriptors
            method (str): Method for training
        """
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.smiles = smiles
        self.fgroups_list = fgroups_list
        self.tokenizer = tokenizer
        self.descriptors = descriptors
        self.method = method

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        smi = self.smiles[idx]  # Get SMILES
        target = self.labels[idx]  # Get label
        x = self._process_smi_(smi)  # Get feature vector
        if self.descriptors:
            descriptors = self._get_descriptors_(smi)  # Get descriptors
            return (
                x,
                descriptors,
                target,
            )  # Return feature vector, descriptors and label
        else:
            return x, target  # Return feature vector and label


class FGRPretrainDataset(BaseDataset):
    def __init__(
        self,
        smiles: List[str],
        fgroups_list: List[MolFromSmarts],
        tokenizer: Tokenizer,
        method: str,
    ) -> None:
        """Pretrain dataset.

        Args:
            smiles (List[str]): List of SMILES
            fgroups_list (List[str]): List of functional groups
            tokenizer (Tokenizer): Pretrained tokenizer
            method (str): Method for training
        """
        self.smiles = smiles
        self.fgroups_list = fgroups_list
        self.tokenizer = tokenizer
        self.method = method

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]  # Get SMILES
        x = self._process_smi_(smi)
        return x
