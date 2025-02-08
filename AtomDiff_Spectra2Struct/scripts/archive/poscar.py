from atomdiff.datasets import StructureDataModule
from atomdiff.models import LitScoreNet

import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

data_module = StructureDataModule(
    data_dir='../data/stem/vasp/*.vasp',
    cutoff=6.0,
    train_prior=True,
    k=0.8,
    train_size=0.9,
    scale_y=1.0,
    dup=64,
    batch_size=16,
    num_workers=4
)

score_net = LitScoreNet(
    num_species=27, 
    num_convs=5, 
    dim=200, 
    ema_decay=0.9999, 
    learn_rate=1e-3
)

# data_module.setup()
# train_loader = data_module.train_dataloader()

trainer = L.Trainer(
    max_steps = 2_500_000,
    logger    = TensorBoardLogger(save_dir='./training_logs/', name='stem-poscar-dup64'),
    callbacks = [TQDMProgressBar(refresh_rate=10)],
)

trainer.fit(
    score_net, data_module
)