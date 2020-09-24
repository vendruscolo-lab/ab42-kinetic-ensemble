import h5py
import numpy as np

from typing import Tuple, Sequence, List, Union, Generator, Callable, Any, Dict, TypeVar, Set
from collections import UserList
from pathlib import Path

T = TypeVar("T")
MaybeListType = Union[List[T], T]
NNDataType = Tuple[List[np.ndarray], np.ndarray]
MaybePathType = Union[Path, str]

FRAMES, DIMENSIONS, FIRST, LAST = 0, 1, 0, -1


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
    
    def get_indices(self, lag: int) -> np.ndarray:
        inds = self._generate_indices(lag)
        allframes = unflatten(np.arange(self.n_points), lengths=self.traj_lengths)
        
        allinds = []
        for i, traj in enumerate(self.data):
            n_points = traj.shape[FRAMES]

            # We'll just skip super short trajectories for now
            if n_points <= lag:
                continue
            
            allinds.append(allframes[inds[i]])
        
        allinds = np.concatenate(allinds)
        eff_len = min(allinds.shape[FRAMES], self.max_frames)
        train_len = int(np.floor(eff_len * self.ratio))
        
        # Reshuffle to remove trajectory level bias
        indices = self._full_indices[self._full_indices < eff_len]
        return allinds[indices][:train_len]
        
    
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
