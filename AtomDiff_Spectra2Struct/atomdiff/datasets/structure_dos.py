import json
import ase, ase.io 
from ase import Atoms
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from tqdm import tqdm
from pathlib import Path
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from scipy.spatial.transform import Rotation

import torch
from torch.nn import functional as F
import numpy as np
import lightning.pytorch as pl

from atomdiff.utils.pbc import periodic_radius_graph

class StructureMPDOSDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        cutoff: float,
        train_size: float = 0.9,
        scale_y: float = 1.0,
        dup: int = 1
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.cutoff = cutoff
        self.scale_y = scale_y
        self.dup = dup

        # read file
        print("Loading JSON file into PyG Data objects...")
        f = open(self.data_dir, 'r')
        data = json.load(f)
        f.close()

        # convert to atoms objects
        atoms_list = []
        dos_list = []
        for d in tqdm(data):
            atoms = Atoms(
                d['atomic_numbers'], 
                positions=d['positions'], 
                cell=np.array(d['cell']), 
                pbc=[True]*3
            )
            atoms_list.append(atoms)
            dos_list.append(d['y'])

        self.atom_encoder = lambda numbers: F.one_hot(numbers - 1, num_classes=100)

        # store list of PyG Data objects as the dataset
        self.dataset = []
        for atoms, dos in tqdm(zip(atoms_list, dos_list)):
            z = torch.tensor(atoms.numbers)
            z = self.atom_encoder(z)
            data = Data(
                z=z.float(),
                pos=torch.tensor(atoms.positions, dtype=torch.float),
                y=torch.tensor(dos, dtype=torch.float),
                cell=np.array(atoms.cell),
                train_mask=torch.rand(len(z)) < train_size,
            )
            self.dataset.append(data)

            # duplicate data
            for _ in range(self.dup):
                self.dataset.append(data.clone())

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data = self.dataset[idx].clone()
        data = self._atomic_graph(data, cutoff=self.cutoff)
        data = self._random_rotate(data)
        return data

    def _atomic_graph(self, data, cutoff):
        cell = torch.tensor(data.cell, dtype=torch.float)
        data.edge_index, edge_vec = periodic_radius_graph(data.pos, cutoff, cell=cell)
        data.edge_len = edge_vec.norm(dim=-1, keepdim=True)
        data.edge_attr = torch.hstack([edge_vec, data.edge_len])
        return data
    
    def _random_rotate(self, data):
        R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float, device=data.pos.device)
        data.pos @= R
        data.edge_attr[:, :3] @= R
        return data

class StructureMPDOSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        cutoff: float,
        scale_y: float = 1.0,
        dup: int = 1,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.cutoff = cutoff
        self.scale_y = scale_y
        self.dup = dup
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        self.train_set = StructureMPDOSDataset(
            data_dir=self.data_dir,
            cutoff=self.cutoff,
            scale_y=self.scale_y,
            dup=self.dup
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def teardown(self, stage=None):
        pass