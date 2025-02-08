from atomdiff.models.prior import LitScoreNet
from atomdiff.datasets import *

import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

data_module = StructureMPDataModule(
    data_dir='data/perov_5_supercell/',
    cutoff=6.0,
    train_prior=True,
    k=0.8,
    train_size=0.9,
    scale_y=1.0,
    dup=1,
    batch_size=8,
    num_workers=4
)

score_net = LitScoreNet(
    num_species=56, 
    num_convs=5, 
    dim=200, 
    ema_decay=0.9999, 
    learn_rate=1e-3
)

# data_module.setup()
# train_loader = data_module.train_dataloader()

trainer = L.Trainer(
    max_steps = 1_420_000,
    logger    = TensorBoardLogger(save_dir='./training_logs/', name='perov-5-supercell-dup1'),
    callbacks = [TQDMProgressBar(refresh_rate=10)],
)

trainer.fit(
    score_net, data_module
)