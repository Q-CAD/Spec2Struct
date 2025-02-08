import json
import ase.io
from ase import Atoms
from glob import glob
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
from atomdiff.diffusers import VarianceExplodingDiffuser


class StructureDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        cutoff: float,
        k: float,
        train_prior: bool = True,
        train_size: float = 0.9,
        scale_y: float = 1.0,
        dup: int = 1,
        fractional_coors: bool = False
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.cutoff = cutoff
        self.train_prior = train_prior
        self.train_size = train_size
        self.scale_y = scale_y
        self.dup = dup
        self.diffuser = VarianceExplodingDiffuser(k=k)

        print("Getting PyG Data objects...")
        atoms_list = self.get_atoms_list()

        # onehot encoder for atom type
        self.atom_encoder = lambda numbers: F.one_hot(numbers - 1, num_classes=100)

        # store list of PyG Data objects as the dataset
        self.dataset = []
        for atoms in tqdm(atoms_list):
            # one-hot encode atomic numbers
            z = torch.tensor(atoms.numbers)
            z = self.atom_encoder(z)

            # scale positions if needed
            if fractional_coors:
                positions = torch.tensor(atoms.get_scaled_positions(), dtype=torch.float)
            else:
                positions = torch.tensor(atoms.positions, dtype=torch.float)

            # create PyG Data object
            data = Data(
                z=z.float(),
                pos=positions,
                train_mask=torch.rand(len(z)) < train_size,
                cell=np.array(atoms.cell)
            )

            # duplicate data
            for _ in range(self.dup):
                self.dataset.append(data.clone())

    def get_atoms_list(self):
        if self.data_dir.endswith('.json'):
            f = open(self.data_dir, 'r')
            data = json.load(f)
            f.close()

            atoms_list = []
            for d in tqdm(data):
                atoms = Atoms(
                    d['atomic_numbers'], 
                    positions=d['positions'], 
                    cell=np.array(d['cell']), 
                    pbc=[True]*3
                )
                atoms_list.append(atoms)

            return atoms_list
        elif Path(self.data_dir).is_dir():
            cif_files = sorted(Path(self.data_dir).glob('*.cif'))
            vasp_files = sorted(Path(self.data_dir).glob('*.vasp'))
            X_fnames = cif_files + vasp_files

            atoms_list = []
            for f in tqdm(X_fnames):
                atoms_list.append(ase.io.read(f))
            return atoms_list

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data = self.dataset[idx].clone()
        if self.train_prior:
            data = self._diffuse_pos(data)
        data = self._atomic_graph(data, cutoff=self.cutoff)
        data = self._random_rotate(data)
        return data
    
    def _diffuse_pos(self, data):
        data.t = torch.rand(1).clip(self.diffuser.t_min, self.diffuser.t_max).expand(data.pos.size(0), 1)
        data.pos, data.eps_r = self.diffuser.forward_noise(data.pos, data.t)
        data.sigma_r = self.diffuser.sigma(data.t)
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
        if self.train_prior:
            data.eps_r @= R
        return data
        
class StructureDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        cutoff: float,
        k: float,
        train_prior: bool = True,
        train_size: float = 0.9,
        scale_y: float = 1.0,
        dup: int = 1,
        fractional_coors: bool = False,
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.cutoff = cutoff
        self.k = k
        self.train_prior = train_prior
        self.train_size = train_size
        self.scale_y = scale_y
        self.dup = dup
        self.fractional_coors = fractional_coors
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_set = StructureDataset(
            self.data_dir,
            self.cutoff,
            self.k,
            self.train_prior,
            self.train_size,
            self.scale_y,
            self.dup,
            self.fractional_coors
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def teardown(self, stage=None):
        pass