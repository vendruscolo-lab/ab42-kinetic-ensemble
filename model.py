#!/usr/bin/env python
# coding: utf-8

# Training params
ratio = 0.9          # Train-Test split ratio
attempts = 20        # Number of times to run
width = 256
depth = 5
learning_rate = 5e-2
dropout = 0.0
regularization = 1e-8


# # Neural network
# 
# In this notebook we set up the neural networks with VAMPNet scoring functions and train them for different output sizes and estimate errors by bootstrap aggregation. This notebook can be used with `papermill` to run all cells automatically with given parameters. We first define the imports and useful utility functions.

import gc
from glob import glob
from multiprocessing import Pool
import itertools
import os
from typing import List, Tuple, Sequence
import warnings

import h5py
import mdtraj as md
import numpy as np
import pyemma as pe
from scipy.linalg import eig

import vampnet
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Dense, Activation, Flatten, Input, BatchNormalization,
                                     concatenate, Dropout, AlphaDropout, Layer)
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, Callback
import tensorflow.keras.backend as K

tf.compat.v1.disable_eager_execution()
warnings.filterwarnings('ignore')
vamp = vampnet.VampnetTools(epsilon=1e-7)

# ## Utility functions
# 
# The version of Keras we're using unfortunately doesn't have `restore_best_weights` implemented, so I copied this from a newer version.

# In[3]:


class EarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs that produced the monitored
            quantity with no improvement after which training will
            be stopped.
            Validation quantities may not be produced for every
            epoch, if the validation frequency
            (`model.fit(validation_freq=5)`) is greater than one.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value


# In[4]:


def unflatten(source: np.ndarray, lengths: List[int]) -> List[np.ndarray]:
    """
    Takes an array and returns a list of arrays.
    
    Parameters
    ----------
    source
        Array to be unflattened.
    lengths
        List of integers giving the length of each subarray.
        Must sum to the length of source.
    
    Returns
    -------
    unflat
        List of arrays.
    
    """
    conv = []
    lp = 0
    for arr in lengths:
        arrconv = []
        for le in arr:
            arrconv.append(source[lp:le + lp])
            lp += le
        conv.append(arrconv)
    ccs = list(itertools.chain(*conv))
    return ccs


# In[5]:


def sort_lengths(flatlengths: Sequence[int], shapes: Sequence[int]) -> List[List[int]]:
    """
    Takes a list of lengths and returns a list of lists of lengths.
    
    Parameters
    ----------
    flatlengths
        List of lengths
    shapes
        List of shapes
    
    Returns
    -------
    lengths
        List of lists of lengths
    
    """
    lengths = []
    i = 0
    for n in shapes:
        arr = []
        for _ in range(n):
            arr.append(flatlengths[i])
            i += 1
        lengths.append(arr)
    return lengths


# In[6]:


def triu_inverse(x: np.ndarray, n: int) -> np.ndarray:
    """
    Converts flattened upper-triangular matrices into full symmetric matrices.
    
    Parameters
    ----------
    x
        Flattened matrices
    n
        Size of the n * n matrix
    
    Returns
    -------
    mat
        Array of shape (length, n, n)
    
    """
    length = x.shape[0]
    mat = np.zeros((length, n, n))
    a, b = np.triu_indices(n, k=1)
    mat[:, a, b] = x
    mat += mat.swapaxes(1, 2)
    return mat


# In[8]:


class VAMPu(Layer):
    def __init__(self, units, activation, **kwargs):
        self.M = units
        self.activation = activation
        
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        self.u_kernel = self.add_weight(name="u_var", shape=(self.M, ), trainable=True,
                                        initializer=tf.constant_initializer(1. / self.M))
        super().build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return [self.M] * 2 + [(self.M, self.M)] * 4 + [self.M]
    
    def _tile(self, x, n_batch):
        x_exp = tf.expand_dims(x, axis=0)
        shape = x.get_shape().as_list()
        return tf.tile(x_exp, [n_batch, *([1] * len(shape))])
    
    def call(self, x):
        chi_t, chi_tau = x
        n_batch = tf.shape(chi_t)[0]
        norm = 1. / tf.cast(n_batch, dtype=tf.float32)
        
        corr_tau = norm * tf.matmul(chi_tau, chi_tau, transpose_a=True)
        chi_mean = tf.reduce_mean(chi_tau, axis=0, keepdims=True)
        kernel_u = tf.expand_dims(self.activation(self.u_kernel), axis=0)
        
        u = kernel_u / tf.reduce_sum(chi_mean * kernel_u, keepdims=True)
        v = tf.matmul(corr_tau, u, transpose_b=True)
        mu = norm * tf.matmul(chi_tau, u, transpose_b=True)
        sigma = tf.matmul(chi_tau * mu, chi_tau, transpose_a=True)
        gamma = chi_tau * tf.matmul(chi_tau, u, transpose_b=True)
        
        C00 = norm * tf.matmul(chi_t, chi_t, transpose_a=True)
        C11 = norm * tf.matmul(gamma, gamma, transpose_a=True)
        C01 = norm * tf.matmul(chi_t, gamma, transpose_a=True)
        
        return [
            self._tile(var, n_batch) for var in (u, v, C00, C11, C01, sigma)
        ] + [mu]


# In[9]:


class VAMPS(Layer):
    def __init__(self, units, activation, order=20, renorm=False, **kwargs):
        self.M = units
        self.activation = activation
        self.renorm = renorm
        self.order = order
        
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        self.S_kernel = self.add_weight(name="S_var", shape=(self.M, self.M), trainable=True,
                                        initializer=tf.constant_initializer(0.1))
        super().build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return [(self.M, self.M)] * 2 + [self.M] + [(self.M, self.M)]
    
    def call(self, x):
        if len(x) == 5:
            v, C00, C11, C01, sigma = x
        else:
            chi_t, chi_tau, u, v, C00, C11, C01, sigma = x
            u = u[0]
        
        n_batch = tf.shape(v)[0]
        norm = 1. / tf.cast(n_batch, dtype=tf.float32)
        C00, C11, C01 = C00[0], C11[0], C01[0]
        sigma, v = sigma[0], v[0]
        
        kernel_w = self.activation(self.S_kernel)
        w1 = kernel_w + tf.transpose(kernel_w)
        w_norm = w1 @ v
        
        # Numerical problems with using a high p-norm
        if self.renorm:
            quasi_inf_norm = lambda x: tf.reduce_max(tf.abs(x))
            w1 = w1 / quasi_inf_norm(w_norm)
            w_norm = w1 @ v
        
        w2 = (1 - tf.squeeze(w_norm)) / tf.squeeze(v)
        S = w1 + tf.linalg.diag(w2)
        
        if len(x) == 8:
            q = (norm * tf.transpose(tf.matmul(S, chi_tau, transpose_b=True))
                 * tf.matmul(chi_tau, u, transpose_b=True))
            probs = tf.reduce_sum(chi_t * q, axis=1)
        
        K = S @ sigma
        vamp_e = tf.transpose(S) @ C00 @ S @ C11 - 2 * tf.transpose(S) @ C01
        vamp_e_tile = tf.tile(tf.expand_dims(vamp_e, axis=0), [n_batch, 1, 1])
        K_tile = tf.tile(tf.expand_dims(K, axis=0), [n_batch, 1, 1])
        S_tile = tf.tile(tf.expand_dims(S, axis=0), [n_batch, 1, 1])
        
        if len(x) == 5:
            return [vamp_e_tile, K_tile, tf.zeros((n_batch, self.M)), S_tile]
        else:
            return [vamp_e_tile, K_tile, probs, S_tile]


# In[10]:


def matrix_inverse(mat):
    """
    Calculates the inverse of a square matrix.
    
    Parameters
    ----------
    mat
        Square real matrix
    
    Returns
    -------
    inv
        Inverse of the matrix
    
    """
    eigva, eigveca = np.linalg.eigh(mat)
    inc = eigva > epsilon
    eigv, eigvec = eigva[inc], eigveca[:, inc]
    return eigvec @ np.diag(1. / eigv) @ eigvec.T

def covariances(data):
    """
    Calculates (lagged) covariances.
    
    Parameters
    ----------
    data
        Data at time t and t + tau
    
    Returns
    -------
    C0inv
        Inverse covariance
    Ctau
        Lagged covariance
    
    """
    chil, chir = data
    norm = 1. / chil.shape[0]
    C0, Ctau = norm * chil.T @ chil, norm * chil.T @ chir
    C0inv = matrix_inverse(C0)
    return C0inv, Ctau

def _compute_pi(K):
    """
    Calculates the stationary distribution of a transition matrix.
    
    Parameters
    ----------
    K
        Transition matrix
    
    Returns
    -------
    pi
        Normalized stationary distribution
    
    """
    eigv, eigvec = np.linalg.eig(K.T)
    pi_v = eigvec[:, ((eigv - 1) ** 2).argmin()]
    return pi_v / pi_v.sum(keepdims=True)


# In[11]:


from typing import Tuple, Sequence, List, Union, Generator, Callable, Any, Dict, TypeVar, Set
from collections import UserList
from pathlib import Path

T = TypeVar("T")
MaybeListType = Union[List[T], T]
NNDataType = Tuple[List[np.ndarray], np.ndarray]
MaybePathType = Union[Path, str]

FRAMES, DIMENSIONS, FIRST, LAST = 0, 1, 0, -1


# In[12]:


def make_list(item: MaybeListType[T], cls=list) -> List[T]:
    """
    Turn an object into a list, if it isn't already.
    
    Parameters
    ----------
    item
        Item to contain in a list
    
    Returns
    -------
    list
        List with item as only element
    
    """
    if not isinstance(item, list):
        item = [item]
    return cls(item)


# In[13]:


class DataSet:
    def __init__(self, trains: List[np.ndarray], valids: List[np.ndarray]=None,
                 y_train: np.ndarray=None, y_valid: np.ndarray=None):
        """
        DataSet - Container for training and validation data.
        
        Parameters
        ----------
        trains
            List of training datasets
        valids
            List of validation datasets
        y_train
            Dummy training target data
        y_valid
            Dummy validation target data
        
        """
        self.trains = trains
        self.valids = valids
        self.y_train = y_train
        self.y_valid = y_valid
    
    def __len__(self) -> int:
        return self.n_train
    
    def __getitem__(self, key: int) -> "DataSet":
        if isinstance(key, int):
            data = self.__class__([t[key][None] for t in self.trains],
                                  [t[key][None] for t in self.valids])
        else:
            data = self.__class__([t[key] for t in self.trains],
                                  [t[key] for t in self.valids])
        if self.n is not None:
            data.n = self.n
        return data
    
    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return self.trains[FIRST].shape[FRAMES]
    
    @property
    def n_valid(self) -> int:
        """Number of validation samples."""
        return self.valids[FIRST].shape[FRAMES]
    
    @property
    def n_dims(self) -> int:
        """Number of input dimensions."""
        return self.trains[FIRST].shape[DIMENSIONS]
    
    @property
    def n(self) -> int:
        """Number of output dimensions."""
        if self.y_train is None:
            return None
        return self.y_train.shape[DIMENSIONS]
    
    @n.setter
    def n(self, n: int):
        self.y_train = np.zeros((self.n_train, n))
        self.y_valid = np.zeros((self.n_valid, n))
    
    @property
    def train(self) -> NNDataType:
        """Training and target data pair."""
        return self.trains, self.y_train
    
    @property
    def valid(self) -> NNDataType:
        """Validation and target data pair."""
        return self.valids, self.y_valid


# In[14]:


class DataGenerator:
    def __init__(self, data: MaybeListType[np.ndarray],
                 ratio: float=0.9, dt: float=1.0, max_frames: int=None):
        """
        DataGenerator - Produces data for training a Koopman model.
        
        Parameters
        ----------
        data
            Input data as (a list of) ndarrays with
            frames as rows and features as columns
        ratio
            Train / validation split ratio
        dt
            Timestep of the underlying data
        max_frames
            The maximum number of frames to use
        
        """
        self._data = make_list(data)
        self.ratio = ratio
        self.dt = dt
        self.max_frames = max_frames or self.n_points
        
        # Generate lag = 0 indices, we will use these for different
        # lag times later. That way we can retrain with essentially
        # the same data for different lag times.
        self.regenerate_indices()
    
    @property
    def data(self) -> List[np.ndarray]:
        return self._data
    
    @property
    def n_dims(self) -> int:
        """Number of dimensions in the input data."""
        return self.data[FIRST].shape[DIMENSIONS]
    
    @property
    def n_points(self) -> int:
        """Number of frames in the input data."""
        return sum(self.traj_lengths)
    
    @property
    def n_traj(self) -> int:
        """Number of trajectories in the input data."""
        return len(self.data)
    
    @property
    def traj_lengths(self) -> int:
        """Length of all trajectories in the input data."""
        return [len(t) for t in self.data]
    
    @property
    def data_flat(self) -> np.ndarray:
        """The flattened input data."""
        return np.vstack(self.data)
    
    @classmethod
    def from_state(cls, data: MaybeListType[np.ndarray],
                   filename: MaybePathType) -> "DataGenerator":
        """
        Creates a DataGenerator object from previously saved index data.
        
        Parameters
        ----------
        data
            Input data as (a list of) ndarrays with
            frames as rows and features as columns
        filename
            File to load the indices from.
        
        """
        gen = cls(data)
        gen.load(filename)
        
        # Check for data consistency
        assert gen.n_traj == len(data), "Inconsistent data lengths!"
        assert all(len(gen._indices[i]) == gen.traj_lengths[i]
                   for i in range(gen.n_traj)), "Inconsistent trajectory lengths!"
        return gen
    
    def regenerate_indices(self):
        """Regenerate random indices."""
        # We use a dict here because we might otherwise desync
        # our indices and trajectories when generating the 
        # train and test data. This way we're sure we're
        # accessing the correct indices.
        self._indices = {}
        for i, traj in enumerate(self.data):
            inds = np.arange(traj.shape[FRAMES])
            np.random.shuffle(inds)
            self._indices[i] = inds
        
        # We will also shuffle the whole dataset to avoid
        # preferentially sampling late round trajectories.
        # These are more indices than we will need in practice,
        # because the trajectories are shortened through the
        # lag time. We will just cut out the extra ones later.
        self._full_indices = np.random.choice(
            np.arange(self.max_frames), size=self.max_frames, replace=False)
    
    def truncate_indices(self, index: int):
        """
        Truncate the indices up to a maximum entry.
        Useful for generating convergence data.
        
        Parameters
        ----------
        index
            Maximum index to use
        
        """
        for idx in self._full_indices[self._full_indices > index]:
            del self._indices[idx]
            
        self._full_indices = self._full_indices[self._full_indices < index]
    
    def save(self, filename: MaybePathType):
        """
        Save the generator state in the form of indices.
        
        Parameters
        ----------
        filename
            File to save the indices to.
        
        """
        with h5py.File(handle_path(filename, non_existent=True), "w") as write:
            # Save the individual trajectory indices
            inds = write.create_group("indices")
            for k, v in self._indices.items():
                inds[str(k)] = v
            
            # Save the indices on a trajectory level
            dset = write.create_dataset("full_indices", data=self._full_indices)
            dset.attrs.update(_get_serializable_attributes(self))
    
    def load(self, filename: MaybePathType):
        """
        Load the generator state from indices.
        
        Parameters
        ----------
        filename
            File to load the indices from.
        
        """
        with h5py.File(handle_path(filename), "r") as read:
            # Object state (ratio etc...)
            self.__dict__.update(read["full_indices"].attrs)
            self._full_indices = read["full_indices"][:]
            
            # All indices
            self._indices = {int(k): v[:] for k, v in read["indices"].items()}
    
    def _generate_indices(self, lag: int) -> Dict[int, np.ndarray]:
        """
        Generates indices corresponding to a particular lag time.
        
        Parameters
        ----------
        lag
            The lag time for data preparation
        
        Returns
        -------
        indices
            Dictionary of trajectory indices with selected frames
        
        """
        indices = {}
        for k, inds in self._indices.items():
            max_points = inds.shape[FRAMES] - lag
            
            # Lag time longer than our trajectory
            if max_points <= 0:
                continue
                
            indices[k] = inds[inds < max_points]
        return indices
    
    def __call__(self, n: int, lag: int) -> DataSet:
        """
        Creates the data for training the neural network.

        Parameters
        ----------
        n
            The size of the output
        lag
            The lag time in steps to be used

        Returns
        -------
        data
            DataSet of training and test data

        """
        xt_shuf = []
        xttau_shuf = []
        indices = self._generate_indices(lag)

        for i, traj in enumerate(self.data):
            n_points = traj.shape[FRAMES]

            # We'll just skip super short trajectories for now
            if n_points <= lag:
                continue

            xt = traj[:n_points - lag]
            xttau = traj[lag:]
            xt_shuf.append(xt[indices[i]])
            xttau_shuf.append(xttau[indices[i]])

        xt = np.vstack(xt_shuf).astype(np.float32)
        xttau = np.vstack(xttau_shuf).astype(np.float32)

        eff_len = min(xt.shape[FRAMES], self.max_frames)
        train_len = int(np.floor(eff_len * self.ratio))
        
        # Reshuffle to remove trajectory level bias
        inds = self._full_indices[self._full_indices < eff_len]
        xt, xttau = xt[inds], xttau[inds]

        return DataSet(
            [xt[:train_len], xttau[:train_len]],
            [xt[train_len:eff_len], xttau[train_len:eff_len]],
            np.zeros((train_len, 2 * n), dtype=np.float32),
            np.zeros((eff_len - train_len, 2 * n), dtype=np.float32))


# In[15]:


class KeepLast(UserList):
    def __init__(self, data: Sequence[T]):
        """
        Constructs a list that will always keep the first item.
        
        Parameters
        ----------
        data
            Data to construct the list from
        
        """
        self.data = list(reversed(data))
    
    def pop_first(self) -> T:
        """
        Returns the first item from the list, but only deletes
        it if there's at least one more item in the list.
        
        Returns
        -------
        item
            First item
        
        """
        if len(self) < 2:
            return self.data[LAST]
        return self.data.pop(LAST)


# In[16]:


def _split(data: np.ndarray, axis=LAST) -> List[np.ndarray]:
    """
    Utility function for splitting the output from two network lobes.
    
    Parameters
    ----------
    data
        Array to split
    axis
        Axis to split along
    
    Returns
    -------
    split
        2 arrays of half width
    
    """
    n = data.shape[axis] // 2
    return [data[:, :n], data[:, n:]]

def handle_path(path: MaybePathType, non_existent: bool=False) -> Path:
    """
    Check path validity and return `Path` object.

    Parameters
    ----------
    path
        Filepath to be checked.
    non_existent
        If false, will raise an error if the path does not exist.

    Returns
    -------
    path
        The converted and existing path.

    """
    if not isinstance(path, Path):
        try:
            path = Path(path)
        except Exception as err:
            message = "Couldn't read path {0}! Original message: {1}"
            raise ValueError(message.format(path, err))
    if not path.exists() and not non_existent:
        raise IOError("File {0} does not exist!".format(path))
    if not path.parent.exists():
        path.parent.mkdir()
    return path


# In[17]:


VALIDS = {int, float, str, list}
def _get_serializable_attributes(obj: object) -> Dict[str, Any]:
    """
    Finds all object attributes that are serializable with HDF5.
    
    Parameters
    ----------
    obj
        Object to serialize
    
    Returns
    -------
    attributes
        All serializable public attributes
    
    """
    return {k: v for k, v in obj.__dict__.items()
            if any(isinstance(v, valid) for valid in VALIDS)
            and not k.startswith("_")}


# In[18]:


class NNModel:
    def __init__(self, model: Union[Model, Layer],
                 loss: MaybeListType[Callable[..., Any]]=None,
                 metric: MaybeListType[Callable[..., Any]]=None,
                 learning_rate: MaybeListType[float]=None,
                 batchsize: int=5000,
                 epochs: int=100,
                 callback: Callback=None,
                 save_initial_weights=False,
                 verbose: int=0):
        """
        Neural network model interface class for Keras.
        
        Parameters
        ----------
        loss
            Loss function(s) to use, if more than one is specified
            subsequent calls to `compile` or `compile_fit` will use
            the next loss function.
        metric
            Fitting metric(s) to use
        learning_rate
            Learning rate for the Adam optimizer. Will accept
            multiple learning rates like `loss`.
        batchsize
            Batchsize to use for training and testing
        epochs
            The maximum number of epochs while training
        callback
            The callback to use while training
        save_initial_weights
            Whether to save the initial weights for possible later restoration
        verbose
            The verbosity level for Keras
        
        Attributes
        ----------
        n_opt
            The number of times `compile_fit` will be called, given
            potentially multiple loss functions and / or learning rates.
        
        """
        self._model = model
        self._history = []
        
        # Using a special list structure allows to use
        # a varying number of compile parameters
        self.loss = make_list(loss, cls=KeepLast)
        self.learning_rate = make_list(learning_rate, cls=KeepLast)
        self.n_opt = max(len(d) for d in (self.loss, self.learning_rate))
        
        self.metric = metric
        self.callback = make_list(callback)
        self.batchsize = batchsize
        self.epochs = epochs
        self.verbose = verbose
        self.save_initial_weights = save_initial_weights
        self._initial_weights = self.weights if self.save_initial_weights else None
    
    @property
    def weights(self) -> List[np.ndarray]:
        """The weights of the model layers."""
        return self._model.get_weights()
    
    @weights.setter
    def weights(self, weights: MaybeListType[np.ndarray]):
        self._model.set_weights(weights)
    
    def reset_weights(self):
        """Reset the weights to the initialized state."""
        if not self.save_initial_weights:
            raise ValueError("Initial weights were not saved!")
        self.weights = self._initial_weights
    
    def compile(self):
        """Compiles the neural network model with the parameters set at instantiation."""
        self._model.compile(
            optimizer=Adam(learning_rate=self.learning_rate.pop_first(),
                           epsilon=0.0001, clipnorm=1.0, beta_1=0.99),
            loss=self.loss.pop_first(),
            metrics=self.metric)
    
    def fit(self, train: NNDataType, valid: NNDataType, **kwargs):
        """
        Trains the neural network with the specified training data.
        
        Parameters
        ----------
        train
            Training data in the form of a list of inputs and a target output.
        valid
            Validation data in the form of a list of inputs and a target output.
        kwargs
            Additional parameters for Keras `fit` or to override object parameters.
        
        """
        # Pass the defaults specified in the constructor and
        # optionally override them with explicitly passed options.
        batchsize = train[-1].shape[0] if self.batchsize == -1 else self.batchsize
        options = dict(batch_size=batchsize, epochs=self.epochs,
                       verbose=self.verbose, callbacks=self.callback, shuffle=True)
        
        # We have to filter out None because we would like to
        # use the Keras defaults if at all possible.
        options = {k: v for k, v in options.items() if v is not None}
        options.update(kwargs)
        hist = self._model.fit(*train, validation_data=valid, **options)
        self._history.append(hist)
    
    def compile_fit(self, train: NNDataType, valid: NNDataType, **kwargs):
        """
        Compiles and trains the neural network with the specified training data
        repeatedly, until all loss functions or learning rates have been used.
        
        Parameters
        ----------
        train
            Training data in the form of a list of inputs and a target output.
        valid
            Validation data in the form of a list of inputs and a target output.
        kwargs
            Additional parameters for Keras `fit` or to override object parameters.
        
        """
        for _ in range(self.n_opt):
            self.compile()
            hist = self.fit(train, valid, **kwargs)
            self._history.append(hist)
        
    def predict(self, data: MaybeListType[np.ndarray]) -> List[np.ndarray]:
        """
        Projects data through the neural network.
        
        Parameters
        ----------
        data
            Input data to predict
        
        Returns
        -------
        predicted
            Predicted data
        
        """
        # This is the case when we're predicting the state
        # assignments of a biased trajectory
        if isinstance(data, np.ndarray):
            data = [data, data]
        elif len(data) == 1:
            data = [data[0], data[0]]
            
        # Specifying batchsize is crucial because
        # we have a batch normalization layer in our model!
        return self._model.predict(data, batch_size=self.batchsize)
    
    def predict_dataset(self, data: DataSet, valid: bool=False,
                        func: Callable[..., Any]=None) -> DataSet:
        """
        Projects a DataSet through the neural network.
        
        Parameters
        ----------
        data
            Input DataSet to predict
        valid
            Whether to also process any validation data in the passed data object
        func
            Optional function to process the raw output of the neural network
        
        Returns
        -------
        predicted
            Predicted data as a DataSet
        
        """
        train_pred = self.predict(data.trains)
        valid_pred = self.predict(data.valids) if valid else None
        
        # We will sometimes need to process the output before
        # forming a DataSet, e.g. splitting the network lobes
        if func is not None:
            train_pred = func(train_pred)
            if valid_pred is not None:
                valid_pred = func(valid_pred)
        return DataSet(train_pred, valid_pred)
        
    def evaluate(self, data: NNDataType) -> float:
        """
        Evaluates the score by passing the output to the specified loss function.
        
        Parameters
        ----------
        data
            Data to use for evaluation, typically validation data
        
        Returns
        -------
        score
            The loss function score
        
        """
        return self._model.evaluate(*data, batch_size=self.batchsize, verbose=self.verbose)
    
    # TODO: Newer Keras versions should be able to save directly to an HDF5 group
    def save(self, group: h5py.Group):
        """
        Save the model to a file.
        
        Parameters
        ----------
        filename
            Path to an HDF5 file
        
        """
        for i, weight in enumerate(self._model.get_weights()):
            group[str(i)] = weight
    
    def load(self, group: h5py.Group):
        """
        Load the model from a file.
        
        Parameters
        ----------
        filename
            Path to an HDF5 file
        
        """
        n_items = len(group)
        self._model.set_weights([
            group[str(i)] for i in range(n_items)
        ])
    
    def load_weights(self, filename: MaybePathType):
        """
        Load the model weights from a file.
        
        Parameters
        ----------
        filename
            Path to an HDF5 file
        
        """
        filename = handle_path(filename).as_posix()
        self._model.load_weights(filename)


# In[19]:


def _build_model(n_input: int, n_output: int, learning_rate: float=1e-4,
                 width: int=1024, depth: int=2, regularization: float=1e-8,
                 dropout: float=0.0, verbose: int=0, batchnorm: bool=False,
                 lr_factor: float=1e-2) -> Dict[str, NNModel]:
    """
    Builds the VAMPNet model.
    
    Parameters
    ----------
    n_input
        Number of input dimensions
    n_output
        Number of output dimensions
    learning_rate
        Learning rate for the chi model
    width
        Width of the layers in neurons
    depth
        Depth of the model in layers
    regularization
        L2 regularization strength per hidden layer
    dropout
        Dropout per hidden layer
    verbose
        Verbosity of the vanilla model
    lr_factor
        Learning rate modifier for the `all` model
    
    Returns
    -------
    models
        Dictionary of all models and the u and S layers
    
    """

    # Input layers
    xti = Input(shape=(n_input,))
    xli = Input(shape=(n_input,))

    # Create hidden layers
    dense = []
    for i in range(depth):
        layer = Dense(
            units=width, activation=KoopmanModel.activation,
            kernel_regularizer=regularizers.l2(regularization),
            kernel_initializer=KoopmanModel.initializer)
        dense.append(layer)
        if i <= depth - 1 and dropout > 0.0:
            dense.append(AlphaDropout(dropout))

    # Optional batch normalization
    bn = BatchNormalization()
    lx = bn(xti) if batchnorm else xti
    rx = bn(xli) if batchnorm else xli
    for i, layer in enumerate(dense):
        lx = dense[i](lx)
        rx = dense[i](rx)

    # Output is softmax for [0, 1] interval
    softmax = Dense(
        units=n_output, activation="softmax",
        kernel_regularizer=regularizers.l2(0.1 * regularization),
        kernel_initializer=KoopmanModel.initializer)
    lx = softmax(lx)
    rx = softmax(rx)

    # Build the model
    merged = concatenate([lx, rx])
    models = {}
    models["chi"] = NNModel(
        Model(inputs=[xti, xli], outputs=merged),
        loss=[vamp._loss_VAMP_sym, vamp.loss_VAMP2_autograd],
        metric=[vamp.metric_VAMP],
        learning_rate=[learning_rate * f for f in (1.0, 0.02)],
        callback=EarlyStopping("val_metric_VAMP", mode="max", min_delta=0.001,
                               patience=5, restore_best_weights=True),
        verbose=0)

    # Auxiliary inputs
    chil_in, chir_in = [Input(shape=(n_output,)) for _ in range(2)]
    v_in = Input(shape=(n_output, 1))
    C00_in, C01_in, C11_in, sigma_in = [
        Input(shape=(n_output, n_output)) for _ in range(4)]

    # Constraint layers
    vlu = VAMPu(n_output, activation=tf.exp)
    vls = VAMPS(n_output, activation=tf.exp, renorm=True)

    # In / Output for full model
    (u_out, v_out, C00_out, C11_out,
     C01_out, sigma_out, mu_out) = vlu([lx, rx])
    Ve_out, K_out, p_out, S_out = vls([
        lx, rx, u_out, v_out, C00_out,
        C11_out, C01_out, sigma_out])

    # In / Output for model with only u and S
    (u_out_b, v_out_b, C00_out_b, C11_out_b,
     C01_out_b, sigma_out_b, _) = vlu([chil_in, chir_in])
    Ve_out_b, K_out_b, p_out_b, S_out_b = vls([
        chil_in, chir_in, u_out_b, v_out_b,
        C00_out_b, C11_out_b, C01_out_b, sigma_out_b])

    # In / Output for new tau prediction model
    Ve_out_s, *_ = vls([v_in, C00_in, C11_in, C01_in, sigma_in])

    # We will need these layers later, so we save them as models
    models["vlu"] = NNModel(vlu, save_initial_weights=True)
    models["vls"] = NNModel(vls, save_initial_weights=True)

    # Build training models, we need to be very careful with batchsizes:
    # https://github.com/keras-team/keras/issues/12400
    early = EarlyStopping("val_loss", patience=10, mode="min", restore_best_weights=True)
    models["all"] = NNModel(
        Model(inputs=[xti, xli], outputs=Ve_out),
        loss=loss_vampe, learning_rate=learning_rate * lr_factor,
        callback=early, epochs=KoopmanModel.n_epoch_aux)
    models["both"] = NNModel(
        Model(inputs=[chil_in, chir_in], outputs=Ve_out_b),
        loss=loss_vampe, learning_rate=5e-4,
        callback=early, epochs=KoopmanModel.n_epoch_aux, batchsize=-1)
    models["S"] = NNModel(
        Model(inputs=[v_in, C00_in, C11_in, C01_in, sigma_in], outputs=Ve_out_s),
        loss=loss_vampe, learning_rate=0.1,
        callback=early, epochs=KoopmanModel.n_epoch_aux, batchsize=-1)

    # Build prediction models
    models["inp"] = NNModel(Model(inputs=[chil_in, chir_in], outputs=[
            v_out_b, C00_out_b, C11_out_b, C01_out_b, sigma_out_b]))
    models["mu"] = NNModel(Model(inputs=[xti, xli], outputs=mu_out))
    models["K"] = NNModel(Model(inputs=[xti, xli], outputs=K_out))
    return models


# In[20]:


class KoopmanModel:
    activation = "selu"
    initializer = "lecun_normal"
    n_epoch_chi = 100
    n_epoch_aux = 10000
    
    def __init__(self, n: int, network_lag: int=4, constrained: bool=True,
                 verbose: int=0, nnargs: Dict[str, Any]=None):
        """
        Provides Koopman model training methods.
        
        Parameters
        ----------
        n
            Network output size
        network_lag
            Training lag for input frame pairs
        constrained
            If we're doing the constrained version for reversibility
        verbose
            Output verbosity
        nnargs
            Arguments passed to the neural network constructor
        
        Attributes
        ----------
        chi_estimated
            True if the VAMPNet part of the model has been estimated
        aux_estimated
            True if the constraint part of the model has been estimated
        generator
            Data generator containing the full dataset
        data
            Data used to train and validate the neural network
        
        """
        self.n_output = n
        self.n_input = None
        self.network_lag = network_lag
        self.verbose = verbose
        self.constrained = constrained
        self.nnargs = nnargs
        self.chi_estimated = False
        self.aux_estimated = False
        
        self._lag = network_lag
        self._reestimated = False
        self._models = None
        self._chi_weights = None
        self._k_cache = {}
        
        # Training and validation data
        self._generator = None
        self.data = None
        
        # Results
        self._K = None
        self._pi = None
        self._mu = None
    
    @classmethod
    def from_file(cls, filename: MaybePathType) -> "KoopmanModel":
        """
        Initialize from a saved model.
        
        Parameters
        ----------
        filename
            HDF5 file with the saved model
        
        """
        koop = cls(n=-1)
        koop.load(handle_path(filename))
        return koop
    
    def __repr__(self) -> str:
        return "<KoopmanModel with shape=({0},{1}), lag={2}, n={3}>".format(
            self.nnargs["width"], self.nnargs["depth"], self.network_lag, self.n_output,
            ", estimated" if self.chi_estimated else "")
    
    def _cleanup(self):
        """
        Cleanup model graph after training. Without this
        the kernel will die due to running out of memory.
        
        """
        gc.collect()
        K.clear_session()
    
    @property
    def n(self) -> int:
        """Alias for `n_output`."""
        return self.n_output
    
    @property
    def is_built(self) -> bool:
        """True if the neural network models have been built."""
        return self._models is not None
        
    @property
    def K(self) -> np.ndarray:
        """The estimated Koopman operator."""
        if self._K is None or self._reestimated:
            self._K = self._models["K"].predict(self.data.trains)[FIRST]
        return self._K
    
    @property
    def mu(self) -> np.ndarray:
        """The estimated mu."""
        if self._mu is None or self._reestimated:
            self._mu = self._models["mu"].predict(self.data.trains).flatten()
        return self._mu
    
    @property
    def pi(self) -> np.ndarray:
        """The estimated equilibrium distribution."""
        return statdist(self.K)
    
    # TODO Add reweighting code here
    @pi.setter
    def pi(self):
        pass
    
    @property
    def dt(self) -> float:
        """The timestep of the underlying data."""
        return self.generator.dt
    
    @property
    def generator(self) -> DataGenerator:
        return self._generator
    
    @generator.setter
    def generator(self, generator: DataGenerator):
        self._generator = generator
        self.n_input = generator.n_dims
        self.data = self.generator(self.n_output, self.network_lag)
    
    @property
    def lag(self) -> int:
        """The model lag time."""
        return self._lag
    
    @lag.setter
    def lag(self, lag: int):
        """
        Update the model lag time for ITS calculation.
        
        Parameters
        ----------
        lag
            Lag time to update the model to
        
        """
        del self.data
        self.data = self.generator(self.n_output, lag)
        self._models["vls"].reset_weights()
        chi_data = self._update_auxiliary_weights(
            optimize_u=False, optimize_S=True, reset_weights=False)
        
        # Prepare training data
        s_data = self._models["inp"].predict_dataset(chi_data, valid=True)[FIRST]
        s_data.n = 1
        
        # Train auxiliary and full model
        self._models["S"].compile_fit(s_data.train, s_data.valid)
        self._models["both"].compile_fit(chi_data.train, chi_data.valid)
        self._lag = lag
        
        # Make sure we recompute any observables
        self._reestimated = True
    
    def estimate_koopman(self, lag: int) -> np.ndarray:
        """
        Estimates the Koopman operator for a given lag time.
        
        Parameters
        ----------
        lag
            Lag time to estimate at
        
        Returns
        -------
        koop
            Koopman operator at lag time `lag`
            
        """
        if lag in self._k_cache:
            return self._k_cache[lag]
        
        self.lag = lag
        K = self.K
        self._k_cache[lag] = K
        return K
    
    def reset_lag(self):
        """Reset the model to the original lag time."""
        self.lag = self.network_lag
    
    def _log(self, msg: str, update: bool=False):
        """Log current status if verbose is set."""
        if self.verbose:
            if update:
                print(msg, end="\r")
            else:
                print(msg)
    
    def _update_auxiliary_weights(self, optimize_u: bool=True, optimize_S: bool=False,
                                  reset_weights: bool=True) -> DataSet:
        """
        Update the weights for the auxiliary model and return new output
        
        Parameters
        ----------
        optimize_u
            Whether to optimize the u vector
        optimize_S
            Whether to optimize the S matrix
        reset_weights
            Whether to reset the weights for the vanilla VAMPNet model
        
        Returns
        -------
        chi
            New training and validation assignments
        
        """
        if reset_weights:
            self._models["chi"].weights = self._chi_weights
        
        # Project training data
        chi_data = self._models["chi"].predict_dataset(self.data, func=_split)

        # Set weights for u vector
        C0inv, Ctau = covariances(chi_data.trains)
        K = C0inv @ Ctau
        if optimize_u:
            pi = _compute_pi(K)
            self._models["vlu"].weights = [np.log(np.abs(C0inv @ pi))]
        
        # Optionally set weights for S matrix
        if optimize_S:
            *_, sigma = self._models["inp"].predict(chi_data.trains)
            sigma_inv = matrix_inverse(sigma[FIRST])
            S_nonrev = K @ sigma_inv
            S_rev = 0.5 * (S_nonrev + S_nonrev.T)
            self._models["vls"].weights = [np.log(np.abs(0.5 * S_rev))]
        
        # Project training and validation data with new weights
        chi_data = self._models["chi"].predict_dataset(self.data, valid=True, func=_split)
        chi_data.n = 2 * self.n_output
        return chi_data
            
    def train_model(self):
        """Train the vanilla VAMPNet model."""
        self._models["chi"].compile_fit(self.data.train, self.data.valid)
        self._chi_weights = self._models["chi"].weights
        self.chi_estimated = True
    
    def train_auxiliary_model(self):
        """Train the auxiliary constraint model."""
        
        # Needs a trained vanilla VAMPNet
        assert self.chi_estimated
        
        # Set initial weights
        self._log("Setting initial auxiliary weights...")
        chi_data = self._update_auxiliary_weights(optimize_S=True)
        
        # Train auxiliary models only
        self._log("Training auxiliary network...")
        self._models["both"].compile_fit(chi_data.train, chi_data.valid)
        
        # Train the whole network
        self._log("Training full network...")
        self._models["all"].compile_fit(self.data.train, self.data.valid)
        self.aux_estimated = True
    
    def train_loop(self):
        """Train the auxiliary constraint model in a loop."""
        # Needs a trained vanilla VAMPNet
        assert self.chi_estimated and self.aux_estimated
        
        score = score_prev = 0.0
        weights = self._models["all"].weights
        
        self._log("Setting up auxiliary training loop...")
        
        self._models["both"].compile()
        self._models["all"].compile()
        
        iteration = 0
        while score >= score_prev:
            chi_data = self._update_auxiliary_weights(reset_weights=False)
            self._models["both"].fit(chi_data.train, chi_data.valid)
            self._models["all"].fit(self.data.train, self.data.valid)
            score = -self._models["all"].evaluate(self.data.valid)
            
            self._log("Iteration {0}: Score: {1}".format(iteration, score))
            
            if score > score_prev:
                score_prev = score
                weights = self._models["all"].weights
            iteration += 1
        
        # Final fit
        self._log("Performing final fit...")
        self._models["all"].weights = weights
        chi_data = self._update_auxiliary_weights(reset_weights=False)
        self._models["both"].fit(chi_data.train, chi_data.valid)

    def save(self, filename: MaybePathType):
        """
        Save the model to a file.
        
        Parameters
        ----------
        filename
            Path to an HDF5 file
        
        """
        filename = handle_path(filename, non_existent=True)
        with h5py.File(filename, "w") as write:
            
            # We need to create a dummy dataset to be able to use `attrs`
            par = write.create_dataset("parameters", (0,))
            parameters = _get_serializable_attributes(self)
            for k, v in parameters.items():
                par.attrs[k] = v
            
            # Save the neural net parameters
            nn = write.create_dataset("nn", (0,))
            for k, v in self.nnargs.items():
                nn.attrs[k] = v
            
            write.create_group("models")
            for k, model in self._models.items():
                model_group = "models/{0}".format(k)
                write.create_group(model_group)
                model.save(write[model_group])
    
    def load(self, filename: MaybePathType):
        """
        Load a model from a file.
        
        Parameters
        ----------
        filename
            Path to an HDF5 file
        
        """
        filename = handle_path(filename)
        with h5py.File(filename, "r") as read:
            # TODO: Put this into a mixin instead
            self.__dict__.update(dict(read["parameters"].attrs))
            self.nnargs = dict(read["nn"].attrs)
            self._models = _build_model(
                self.n_input, self.n_output, verbose=self.verbose, **self.nnargs)
            for k, model in self._models.items():
                model.load(read["models/{0}".format(k)])
    
    def load_chi_weights(self, filename: MaybePathType):
        """
        Load a model from a file.
        
        Parameters
        ----------
        filename
            Path to an HDF5 file
        
        """
        if not self.is_built:
            self._models = _build_model(self.n_input, self.n_output,
                                        verbose=self.verbose, **self.nnargs)
        filename = handle_path(filename)
        self._models["chi"].load_weights(filename)
        self._chi_weights = self._models["chi"].weights
        self.chi_estimated = True
        
    def fit(self, generator: DataGenerator):
        """
        Train the model.
        
        Parameters
        ----------
        generator
            Data generator function, must be callable with output size and lag time
        
        """
        self.generator = generator
        self._models = _build_model(self.n_input, self.n_output,
                                    verbose=self.verbose, **self.nnargs)
        
        self.train_model()
        
        # This is the more compute-intensive part
        if self.constrained:
            self.train_auxiliary_model()
            self.train_loop()
    
    def transform(self, data: MaybeListType[np.ndarray]) -> MaybeListType[np.ndarray]:
        """
        Transform data through the model.
        
        Parameters
        ----------
        data
            Array(s) with input features, no lagged component required
        
        Returns
        -------
        predicted
            Array(s) with probabilistic state assignments
        
        """
        assert self.chi_estimated
        
        # Prediction doesn't care about individual trajectories
        if isinstance(data, list):
            lengths = [len(d) for d in data]
            data = np.vstack(data)
            prob = self._models["chi"].predict(data)
            return unflatten(prob, lengths=lengths)
        
        return self._models["chi"].predict(data)
    
    def fit_transform(self, generator: DataGenerator) -> np.ndarray:
        """
        Fit and transform data through the model.
        
        Parameters
        ----------
        data
            Data generator function, must be callable with output size, lag time,
            and must have a `flat` attribute for the complete input dataset.
        
        Returns
        -------
        predicted
            Array(s) with probabilistic state assignments
        
        """
        self.fit(generator)
        return self.transform(generator.data_flat)
    
    def score(self) -> float:
        """
        Validation score of the trained model.
        
        Returns
        -------
        score
            The VAMP-E score of the fully trained model
        
        """
        assert self.chi_estimated and self.aux_estimated
        return -self._models["all"].evaluate(self.data.valid)
    
    def its(self, lags: Sequence[int]) -> np.ndarray:
        """
        Calculate implied timescales for a sequence of lag times.
        
        Parameters
        ----------
        lags
            Sequence of lag times to calculate the timescales for
        
        Returns
        -------
        its
            Implied timescales for different components
            (dim 0) and lag times (dim 1).
        
        """
        assert self.chi_estimated and self.aux_estimated
        its = np.empty((self.n_output - 1, len(lags)))
        for i, lag in enumerate(lags):
            self._log("Computing {0}/{1}...".format(i + 1, len(lags)), update=True)
            
            K = self.estimate_koopman(lag)
            lambdas = np.linalg.eigvals(K)
            lambdas = np.sort(np.abs(np.real(lambdas)))[:LAST]
            its[:, i] = -lag * self.dt / np.log(lambdas)
        
        self.reset_lag()
        return its
    
    def cktest(self, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the Chapman-Kolmogorov test on the model.
        
        Parameters
        ----------
        steps
            Number of steps to use
        
        Returns
        -------
        est
            Koopman operators estimated at different model lag times
        pred
            Koopman operators predicted at different
            lag times using matrix exponentiation
        
        """
        est = np.empty((self.n, self.n, steps))
        pred = np.empty((self.n, self.n, steps))
        est[:, :, FIRST] = np.eye(self.n)[:self.n]
        pred[:, :, FIRST] = np.eye(self.n)[:self.n]
        
        # Get the current Koopman operator first, because
        # estimating at a new lag time is very expensive
        K = self.K
        
        # Estimate new operators (slow)
        temp_est = np.empty((steps, self.n, self.n))
        temp_est[1] = K
        for nn in range(2, steps):
            self._log("CK test {0}/{1}".format(nn, steps), update=True)
            temp_est[nn] = self.estimate_koopman(self.network_lag * nn)
        self.reset_lag()
        
        # Get new predictions (fast)
        for i in range(self.n):
            vec = np.eye(self.n)[i]
            for nn in range(1, steps):
                est[i, :, nn] = vec @ temp_est[nn]
                pred[i, :, nn] = vec @ np.linalg.matrix_power(K, nn)
        
        return est, pred


# In[21]:


def loss_vampe(y_true, y_pred):
    return tf.linalg.trace(y_pred[FIRST])


# In[22]:


def statdist(X: np.ndarray) -> np.ndarray:
    """
    Calculate the equilibrium distribution of a transition matrix.
    
    Parameters
    ----------
    X
        Row-stochastic transition matrix
    
    Returns
    -------
    mu
        Stationary distribution, i.e. the left
        eigenvector associated with eigenvalue 1.
    
    """
    ev, evec = eig(X, left=True, right=False)
    mu = evec.T[ev.argmax()]
    mu /= mu.sum()
    return mu


