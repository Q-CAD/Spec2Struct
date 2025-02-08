import ase.io
from pathlib import Path
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.transform import Rotation

import torch
import numpy as np
import lightning.pytorch as pl

from atomdiff.utils.pbc import periodic_radius_graph
from atomdiff.diffusers import VarianceExplodingDiffuser


class StructureXANESDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        cutoff: float,
        k: float,
        train_prior: bool = True,
        train_size: float = 0.9,
        scale_y: float = 1.0,
        dup: int = 1
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.cutoff = cutoff
        self.train_prior = train_prior
        self.train_size = train_size
        self.scale_y = scale_y
        self.dup = dup
        self.diffuser = VarianceExplodingDiffuser(k=k)

        # file names to read
        self.X_fnames = sorted(Path(self.data_dir).glob('poscar/train/*.vasp'))
        self.y_fnames = sorted(Path(self.data_dir).glob('xanes/train/*.txt'))

        for i, _ in enumerate(self.X_fnames):
            assert self.X_fnames[i].stem == self.y_fnames[i].stem
        
        # read files
        atoms_list = [ase.io.read(f) for f in self.X_fnames]
        xanes_list = [np.genfromtxt(f) for f in self.y_fnames]

        # wrap atoms outside the periodic boundaries
        for atoms in atoms_list:
            atoms.wrap()

        # onehot encoder for atom type
        unique_numbers = np.concatenate([np.unique(atoms.numbers) for atoms in atoms_list])
        self.atom_encoder = OneHotEncoder(sparse_output=False)
        self.atom_encoder.fit(unique_numbers.reshape(-1, 1))

        # store list of PyG Data objects as the dataset
        self.dataset = []
        for atoms, xanes in zip(atoms_list, xanes_list):
            z = self.atom_encoder.transform(atoms.numbers.reshape(-1, 1))
            data = Data(
                z=torch.tensor(z, dtype=torch.float),
                pos=torch.tensor(atoms.positions, dtype=torch.float),
                y=torch.tensor(xanes * self.scale_y, dtype=torch.float),
                train_mask=torch.rand(len(z)) < train_size,
                cell=np.array(atoms.cell)
            )

            # duplicate data
            for _ in range(self.dup):
                self.dataset.append(data.clone())

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
        
class StructureXANESDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        cutoff: float,
        k: float,
        train_prior: bool = True,
        train_size: float = 0.9,
        scale_y: float = 1.0,
        dup: int = 1,
        batch_size: int = 8,
        num_workers: int = 4
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
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_set = StructureXANESDataset(
            self.data_dir,
            self.cutoff,
            self.k,
            self.train_prior,
            self.train_size,
            self.scale_y,
            self.dup
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