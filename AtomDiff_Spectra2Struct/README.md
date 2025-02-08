# atom-diff
Diffusion over atomic positions. Built from: [graphite](https://github.com/LLNL/graphite).

### Installation
```
conda create -n atomdiff
conda activate atomdiff
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

pip install ase lightning tensorboard

# you might also need to downgrade numpy:
pip install --upgrade numpy==1.26.4
```

Clone and install this repo:

```
git clone https://github.com/shuyijia/atom-diff.git
cd atom-diff/AtomDiff_Spectra2Struct
pip install -e .
```

### Training a Prior
To train a prior (denoising) model, we need to specify the `data_dir` and `run_name`.

For data_dir, you can provide either:

- A directory containing `.cif` or `.vasp` files
- A JSON file with a list of dictionaries, each containing the keys: `positions`, `atomic_numbers`, and `cell`.

For more command line flags, see `train_prior.py`.

#### Example

```
python train_prior.py --data_dir ./data/perov_5/raw_train/ --run_name example_train
```

### Training a Posterior 
To train a posterior (forward) model, please refer to `train_posterior.py`.