# A kinetic ensemble of the Alzheimer's Aβ peptide

This repository contains the full code and some small example data to reproduce our results on the kinetic ensemble of amyloid-β 42. See also the [original implementation](https://github.com/markovmodel/deep_rev_msm) of the constrained VAMPNets.

## Reproducibility information
The analysis was performed on a single Google compute engine instance using 12 vCPUs, 78 GB memory and 1x NVIDIA Tesla V100 GPU. The instance used the `c3-deeplearning-tf-ent-2-1-cu100-20200131` image with CUDA 10.0 and tensorflow 2.1, based on Debian 10. The original environment for this machine is provided in `env-tf-original.yml`. Training a single network takes approximately 2 hours on this architecture.

## Cloud bucket
Full data, including trajectories, backcalculated chemical shifts, neural network weights, intermediate data, and example structures can be accessed through a Google cloud bucket using the [`gsutil` tool](https://cloud.google.com/storage/docs/quickstart-gsutil):

`gsutil -m cp -r gs://kinetic_ensemble_paper/`

The size of the bucket is around 40 GB and includes the following directories:

- `trajectories/`: The simulations trajectories, subsampled to 250 ps timesteps, performed in 5 rounds with 1024 individual trajectories each. The aggregated simulated time is 314 µs. Also includes the chemcial shifts backcalculated with *CamShift* as implemented in Plumed.
- `intermediate/`: Intermediate data files, such as the calculated inter-residue minimum distances, and the full model outputs in the form of transition matrices, weights, and timescales.
- `models/`: The neural network models including weights and trajectory indices used for training.
- `structures{,-alt}/`: State structures sampled from the trajectories sampled two different ways.
- `figs/`: Raw figures for the paper, generated by the notebooks.

## Notebooks
The easiest way to try out the notebooks is by using [`conda`](https://www.anaconda.com/products/individual). We include two environment specifications: `env-tf.yml` specifies the environment to be used for running the neural network, and `env-analysis.yml` specifies the packages needed for the analysis and plotting of the results. Because the installation of tensorflow is mostly highly specific to your machine, we strongly recommend following the [official installation instructions](https://www.tensorflow.org/install). To create the environments, run `conda env create -f env-analysis.yml` and activate the new environment with `conda activate analysis`. You will also need to install a tensorflow 2.* compatible version of [`vamptools`](https://github.com/tlhr/deeptime/tree/master/vampnet) for training.

- `msm-vampe-hyperpar.ipynb`: Hyperparameter search code, can be run with [`papermill`](https://papermill.readthedocs.io/en/latest/) and the `env-tf.yml` environment.
- `msm-vampe-training.ipynb`: Training code, can be run with [`papermill`](https://papermill.readthedocs.io/en/latest/) and the `env-tf.yml` environment.
- `msm-vampe-model-analysis.ipynb`: Simple post-processing of the trained models, includes calculation of the CK-test and the implied timescales. Run with the `env-tf.yml` environment.
- `msm-vampe-analysis.ipynb`: Full analysis and plots of the ensemble, run with the `env-analysis.yml` environment.
- `model.py`: The neural network model code.
