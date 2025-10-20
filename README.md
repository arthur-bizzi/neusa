# Neuro-Spectral Architectures for Causal Physics-Informed Networks

Arthur Bizzi [1,2] (arthur.coutinhobizzi@epfl.ch), Leonardo Moreira [3], Márcio Macedo [2], Leonardo Mendonça [2], Christian Oliveira [2], Vitor Balestro [2],
Lucas Fernandez [2,4], Daniel Yukimura [2], Pavel Petrov [2], João M. Pereira [5], Tiago Novello [2] and Lucas Nissenbaum [2].

[1] École Polytechnique Fédérale de Lausanne (EPFL)<br>
[2] Instituto de Matemática Pura e Aplicada (IMPA) <br>
[3] Universidade do Estado do Rio de Janeiro (UERJ)<br>
[4] Laboratório Nacional de Computação Científica (LNCC) <br>
[5] University of Georgia (UGA)

This is the official repository of the paper "Neuro-Spectral Architectures for Causal Physics-Informed Networks", to appear on NeurIPS 2025 (https://neurips.cc/). You can also check the arXiv version: https://arxiv.org/html/2509.04966v1.

<figure style="text-align:center;">
  <img src="docs/animation.gif" alt="marmousievo" style="display:block; margin:0 auto;">
  <figcaption style="margin-top:0px; font-style: italic;">Wave evolution in a Marmousi medium.</figcaption>
</figure> 

# Setup and Dependencies

The dependencies are managed with Poetry (https://python-poetry.org/). If you don't have Poetry installed yet:
```sh
pip install poetry
```
Then, from the project root:
```sh
poetry install
```
This command will create a virtual environment and install all dependencies listed in `pyproject.toml`.

After installation, you can activate poetry shell (also from the project root):
```sh
poetry shell
```
and run each `.py` file as usual.

If you prefer not to use Poetry, you can install the dependencies listed in `pyproject.toml` manually.

# Training and Evaluating

We implemented NeuSA and baseline models for five equations: Klein-Gordon non-linear with Gaussian initial condition, Klein-Gordon non-linear with triangular initial condition, Burgers 2d, Wave 2d with a three layered medium, Wave 2d with a Marmousi medium and Wave 3d with a two layered medium.

Each equation has two files. One of them implements the NeuSA method, and the other one implements the baseline methods: PINN, FLS (First Layer Sine), QRes and, for some of the equations, PINNsFormer.

When you run one of these scripts, the corresponding model is trained and compared with a ground-truth solution. Some plots are produced as well. The comparison metrics, the trained model and the plots are automatically saved in a subfolder of the `results` directory.

At each script, an argparse provides the following options:

--dir: the name of the subfolder of 'results' directory where the results will be stored;<br>
--seed: the chosen random torch seed;<br>
--device: the device where the code will be executed (a GPU, preferably);<br>
--steps: the number of training steps (we adopted an Adam trainer for all experiments);<br>
--lr: the Adam trainer learning rate;<br>
--model: the baseline model to be trained (except for the NeuSA files).

The default values for the learning rate and for the number of training steps in each script are the ones that we have used to produce the numbers that appear in the paper.

For example, if you run

```sh
python wave2d_marmousi_NeuSA.py --dir my_experiment
```

a NeuSA model for the Wave 2d Marmousi equation will be trained for 5000 steps with learning rate 0.01, since these are the default values in the argparse. The comparison metrics, the trained model and the plots will be saved in the folder `/results/wave2d_marmousi/my_experiment/NeuSA/seed_42`, because 42 is the default seed in the argparse. 

# Citation

Please cite as:

```bibtex
@article{bizzi2025neuro,
  title={Neuro-Spectral Architectures for Causal Physics-Informed Networks},
  author={
    Arthur Bizzi and Leonardo Moreira and Márcio Macedo and 
    Leonardo Mendonça and Christian Oliveira and Vitor Balestro and 
    Lucas Fernandez and Daniel Yukimura and Pavel Petrov and 
    João M. Pereira and Tiago Novello and Lucas Nissenbaum
  },
  journal={arXiv preprint arXiv:2509.04966v1},
  year={2025},
  month={sep},
  eprint={2509.04966v1},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  note={License: CC BY 4.0}
}
```