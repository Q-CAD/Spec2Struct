from atomdiff.datasets import *
from atomdiff.models.forward import LitDOSNet

import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

data_module = StructureMPDOSDataModule(
    data_dir='/net/csefiles/coc-fung-cluster/Shuyi/Datasets/dos/mp_dos_20_train_17386.json',
    cutoff=5.0,
    scale_y=1.0,
    dup=1,
    batch_size=32,
    num_workers=4
)

model = LitDOSNet(
    num_species=100, 
    num_convs=3, 
    dim=256, 
    out_dim=400, 
    learn_rate=1e-5
)

trainer = L.Trainer(
    max_steps = 200_000,
    logger    = TensorBoardLogger(save_dir='./training_logs/', name='dos_net'),
    callbacks = [TQDMProgressBar(refresh_rate=10)],
)

trainer.fit(model, data_module)