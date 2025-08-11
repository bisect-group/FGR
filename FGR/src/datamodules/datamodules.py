import os
from typing import Dict, List, Optional, Union

import pandas as pd
import pyrootutils
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from pytorch_multilabel_balanced_sampler.samplers import ClassCycleSampler
from rdkit.Chem.rdmolfiles import MolFromSmarts
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler

from FGR.src.datamodules.components.global_dicts import TASK_DICT
from FGR.src.datamodules.datasets import FGRDataset, FGRPretrainDataset

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


class FGRDataModule(LightningDataModule):
    """LightningDataModule for single dataset."""

    def __init__(
        self,
        data_dir: str,
        dataset: str,
        method: str,
        descriptors: bool,
        tokenize_dataset: str,
        frequency: int,
        split_type: str,
        fold_idx: int,
        loaders: DictConfig,
    ) -> None:
        """DataModule with standalone train, val and test dataloaders.

        Args:
            data_dir (str): Path to root data folder
            dataset (str): Dataset name
            method (str): Method for training
            descriptors (bool): Whether to use descriptors
            tokenize_dataset (str): Dataset to use for tokenization
            frequency (int): Frequency for tokenization
            split_type (str): Split type
            fold_idx (int): Fold index
            loaders (DictConfig): Loaders config.
        """

        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.method = method
        self.descriptors = descriptors
        self.tokenize_dataset = tokenize_dataset
        self.frequency = frequency
        self.split_type = split_type
        self.fold_idx = fold_idx
        self.regression = TASK_DICT[dataset][2]
        self.num_classes = TASK_DICT[dataset][0]
        self.cfg_loaders = loaders
        self.train_set: Optional[Dataset] = None
        self.valid_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

    def _get_dataset_(
        self,
        split_name: str,
    ) -> Dataset:
        data = pd.read_parquet(
            os.path.join(
                root,
                self.data_dir,
                "tasks",
                self.dataset,
                "splits",
                self.split_type,
                f"fold_{self.fold_idx}",
                f"{split_name}.parquet",
            )
        )  # Read dataset
        smiles = data["SMILES"].astype(str).tolist()  # Get SMILES
        labels = data.drop(columns=["SMILES"]).values  # Get labels

        fgroups = pd.read_parquet(os.path.join(root, self.data_dir, "training", "fg"))[
            "SMARTS"
        ].tolist()  # Get functional groups
        fgroups_list = [MolFromSmarts(x) for x in fgroups]  # Convert to RDKit Mol
        tokenizer = Tokenizer.from_file(
            os.path.join(
                root,
                self.data_dir,
                "training",
                "tokenizers",
                f"BPE_{self.tokenize_dataset}_{self.frequency}.json",
            )
        )  # Load tokenizer
        dataset = FGRDataset(
            smiles,
            labels,
            fgroups_list,
            tokenizer,
            self.descriptors,
            self.method,
        )  # Create dataset
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.train_set`, `self.valid_set`, `self.test_set`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.train_set and not self.valid_set and not self.test_set:
            self.train_set = self._get_dataset_("train")
            self.valid_set = self._get_dataset_("val")
            self.test_set = self._get_dataset_("test")

        if not self.regression:
            labels = self.train_set.labels.int()  # type: ignore
            if self.num_classes > 1:
                self.sampler = ClassCycleSampler(labels=labels)  # type: ignore
                self.cfg_loaders.train.shuffle = False
            else:
                class_weights = 1.0 / torch.tensor(
                    [len(labels[labels == i]) for i in torch.unique(labels)],
                    dtype=torch.float,
                )
                samples_weight = torch.tensor([class_weights[t] for t in labels.int()])
                # Define the sampler
                self.sampler = WeightedRandomSampler(
                    weights=samples_weight,
                    num_samples=len(samples_weight),  # type: ignore
                )
                self.cfg_loaders.train.shuffle = False

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        if not self.regression:
            loader = DataLoader(
                self.train_set,
                sampler=self.sampler,
                **self.cfg_loaders.get("train"),  # type: ignore
            )
        else:
            loader = DataLoader(self.train_set, **self.cfg_loaders.get("train"))  # type: ignore
        return loader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.valid_set, **self.cfg_loaders.get("val"))  # type: ignore

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_set, **self.cfg_loaders.get("test"))  # type: ignore

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass


class FGRPretrainDataModule(LightningDataModule):
    """LightningDataModule for pretraining."""

    def __init__(
        self,
        data_dir: str,
        dataset: str,
        frequency: int,
        method: str,
        loaders: DictConfig,
    ) -> None:
        """DataModule with standalone train and val dataloaders.

        Args:
            data_dir (str): Path to root data folder
            dataset (str): Dataset name
            frequency (int): Frequency for tokenization
            method (str): Method for training
            loaders (DictConfig): Loaders config
        """
        super().__init__()

        self.data_dir = data_dir
        self.dataset = dataset
        self.frequency = frequency
        self.method = method
        self.cfg_loaders = loaders
        self.train_set: Optional[Dataset] = None
        self.valid_set: Optional[Dataset] = None

    def setup(self, stage=None):
        df = pd.read_parquet(
            os.path.join(
                root,
                self.data_dir,
                "training",
                self.dataset,
            )
        )["SMILES"].tolist()  # Get SMILES
        train, valid = train_test_split(
            df, test_size=0.1, random_state=123
        )  # Split into train and validation

        fgroups = pd.read_parquet(os.path.join(root, self.data_dir, "training", "fg"))[
            "SMARTS"
        ].tolist()  # Get functional groups
        fgroups_list = [MolFromSmarts(x) for x in fgroups]  # Convert to RDKit Mol
        tokenizer = Tokenizer.from_file(
            os.path.join(
                root,
                self.data_dir,
                "training",
                "tokenizers",
                f"BPE_{self.dataset}_{self.frequency}.json",
            )
        )  # Load tokenizer

        self.train_set = FGRPretrainDataset(
            train, fgroups_list, tokenizer, self.method
        )  # Create train dataset

        self.valid_set = FGRPretrainDataset(
            valid, fgroups_list, tokenizer, self.method
        )  # Create validation dataset

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_set, **self.cfg_loaders.get("train"))  # type: ignore

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.valid_set, **self.cfg_loaders.get("val"))  # type: ignore

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass
