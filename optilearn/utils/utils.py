import itertools
import os
import random
from glob import glob
from typing import Any, Iterable, Optional

import torch
import yaml
from torch.distributions import uniform


TORCH_PRECISION = torch.float32
TORCH_PRECISION_HIGH = torch.float64
USE_CUDA = True



def load_yml(path_to_file):
    """
    Load a YAML file.

    Args:
        path_to_file (str): The path to the YAML file.

    Returns:
        dict: The loaded YAML file as a dictionary.

    """
    with open(path_to_file) as f:
        file = yaml.safe_load(f)
    return file


def save_to_yml(data, name, save_path=""):
    """
    Save data to a YAML file.

    Args:
        data (Any): The data to be saved.
        name (str): The name of the file.
        save_path (str, optional): The path to save the file. Defaults to "".

    Returns:
        str: The path to the saved file.

    """
    data_path = os.path.join(save_path, name)
    with open(data_path, "w") as f:
        yaml.safe_dump(data, f)
    return data_path


def set_seed_everywhere(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The random seed.

    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device(use_gpu_acceleration=True):
    """
    Get the device for computation.

    Args:
        use_gpu_acceleration (bool, optional): Whether to use GPU acceleration. Defaults to True.

    Returns:
        str: The device for computation.

    """
    device = "cpu"
    if use_gpu_acceleration:
        try:
            if torch.backends.mps.is_available():
                device = "mps"
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        except:
            print("No apple silicone supported")
        if device == "cpu":
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                torch.backends.cudnn.benchmark = True
    return device


def tensor(arr, dtype=TORCH_PRECISION, device=None):
    """
    Convert an array to a PyTorch tensor.

    Args:
        arr: The input array.
        dtype (torch.dtype, optional): The data type of the tensor. Defaults to TORCH_PRECISION.
        device (str, optional): The device to store the tensor. Defaults to None.

    Returns:
        torch.Tensor: The converted tensor.

    """
    if isinstance(arr, torch.Tensor):
        return arr
    if isinstance(arr, list):
        arr = np.array(arr)
    if device is None:
        if torch.cuda.is_available() and USE_CUDA:
            return torch.tensor(arr, dtype=dtype).cuda()
        else:
            return torch.tensor(arr, dtype=dtype)
    else:
        return torch.tensor(arr, device=device, dtype=dtype)


def tensor_to_float(tensor):
    """
    Convert a tensor to a float value.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        float: The converted float value.

    """
    if "cuda" in tensor.device.type:
        tensor = tensor.cpu()
    return float(tensor.detach().numpy())


def tensor_to_numpy(tensor):
    """
    Convert a tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        np.ndarray: The converted NumPy array.

    """
    if "cuda" in tensor.device.type or "mps" in tensor.device.type:
        tensor = tensor.cpu()
    return np.array(tensor)


def get_eval_w(n, dim, range_param=1):
    """
    Get the evaluation weights.

    Args:
        n (int): The number of weights.
        dim (int): The dimension of the weights.
        range_param (int, optional): The range parameter. Defaults to 1.

    Returns:
        list: The list of evaluation weights.

    """
    if n == 0:
        return [[1]]
    min_pref = (1 - range_param) / 2
    max_pref = 0.5 + range_param / 2
    epsilon = 0.00000001
    step = max((max_pref - min_pref) / n, epsilon)
    b = []
    if dim == 4:
        for b1 in np.arange(min_pref, max_pref + epsilon, step):
            b234 = 1 - b1
            for b2 in np.arange(min_pref, b234 + epsilon, step):
                b34 = b234 - b2
                for b3 in np.arange(min_pref, b34 + epsilon, step):
                    b4 = b34 - b3
                    b += [[b1, b2, b3, abs(b4)]]
    elif dim == 5:
        for b0 in np.arange(min_pref, max_pref + epsilon, step):
            b2345 = 1 - b0
            for b1 in np.arange(min_pref, b2345 + epsilon, step):
                b234 = b2345 - b1
                for b2 in np.arange(min_pref, b234 + epsilon, step):
                    b34 = b234 - b2
                    for b3 in np.arange(min_pref, b34 + epsilon, step):
                        b4 = b34 - b3
                        b += [[b0, b1, b2, b3, abs(b4)]]
    elif dim == 6:
        for b1m in np.arange(min_pref, max_pref + epsilon, step):
            b23456 = 1 - b1m
            for b0 in np.arange(min_pref, b23456 + epsilon, step):
                b2345 = b23456 - b0
                for b1 in np.arange(min_pref, b2345 + epsilon, step):
                    b234 = b2345 - b1
                    for b2 in np.arange(min_pref, b234 + epsilon, step):
                        b34 = b234 - b2
                        for b3 in np.arange(min_pref, b34 + epsilon, step):
                            b4 = b34 - b3
                            b += [[b1m, b0, b1, b2, b3, abs(b4)]]
    elif dim == 3:
        for b1 in np.arange(min_pref, max_pref + epsilon, step):
            b234 = 1 - b1
            for b2 in np.arange(min_pref, b234 + epsilon, step):
                b3 = b234 - b2
                b += [[b1, b2, abs(b3)]]
    elif dim == 2:
        # Add code for dim = 2
        pass
        for b1 in np.arange(min_pref, max_pref + epsilon, step):
            b2 = 1 - b1
            b += [[b1, abs(b2)]]
    elif dim == 1:
        b = [[1]]
    return np.array(b)


import numpy as np


def pareto_filter(y):
    """
    Filters a set of points based on the Pareto dominance criterion.

    Args:
        y (numpy.ndarray): An array of shape (n_samples, n_objectives) representing the points to be filtered.

    Returns:
        tuple: A tuple containing two arrays:
            - A numpy.ndarray of shape (n_pareto_points, n_objectives) representing the filtered points.
            - A numpy.ndarray of shape (n_pareto_points,) representing the indices of the filtered points in the original array.

    """
    is_pareto = [~((i < y).all(axis=1).any()) and ~((i == y).any(axis=1) & (i < y).any(axis=1)).any() for i in y]
    return np.unique(y[is_pareto], axis=0), np.arange(len(y))[is_pareto]


def calc_hypervolume(y, utopia, antiutopia, n_samples=10000, rnd=True):
    """
    Calculates the hypervolume indicator for a given set of solutions.

    Args:
        y (array-like): The solutions to evaluate.
        utopia (array-like): The utopia point.
        antiutopia (array-like): The antiutopia point.
        n_samples (int, optional): The number of samples to use for Monte Carlo integration. Defaults to 10000.
        rnd (bool, optional): Whether to use random sampling or linearly spaced sampling. Defaults to True.

    Returns:
        float: The hypervolume indicator.

    """
    if rnd:
        dist = uniform.Uniform(tensor(antiutopia), tensor(utopia))
        p = dist.sample([n_samples])
    else:
        p = tensor(multilinspace(utopia, antiutopia, n_samples))
    p_expand = p.unsqueeze(2).permute(2, 1, 0)
    y_expand = tensor(y.astype(float)).unsqueeze(2)
    return (p_expand < y_expand).all(dim=1).any(dim=0).sum() / float(n_samples)
    # return HV(ref_point=antiutopia)(-y)


def multilinspace(a, b, n):
    """
    Linspace for multi-dimensional intervals.

    Generates a multi-dimensional linspace between two points `a` and `b` with `n` points.

    Args:
        a (list or array): The starting point of the interval.
        b (list or array): The ending point of the interval.
        n (int): The number of points to generate.

    Returns:
        array: A multi-dimensional array containing the generated points.
    """
    dim = len(a)
    if dim == 1:
        return np.linspace(a, b, n)

    n = int(np.floor(n ** (1.0 / dim)))
    tmp = []
    for i in range(dim):
        tmp.append(np.linspace(a[i], b[i], n))
    x = np.meshgrid(*tmp)
    y = np.zeros((n**dim, dim))
    for i in range(dim):
        y[:, i] = x[i].flatten()
    return y


def calc_opt_reward(prefs, front, u_func=None):
    """
    Calculates the optimal reward for a given set of preferences and a front.

    Args:
        prefs (numpy.ndarray): The preferences matrix.
        front (numpy.ndarray): The front matrix.
        u_func (function, optional): The utility function to calculate rewards. Defaults to None.

    Returns:
        numpy.ndarray: The optimal reward matrix.

    """
    if u_func is None:
        u_func = lambda x, y: (x * y).sum(axis=1)
    prefs = np.float32(prefs)
    w_front = np.zeros(prefs.shape)
    for n, w in enumerate(prefs):
        id = u_func(front, w).argmax()
        w_front[n] = front[id]
    return w_front


def list_all_files(path: str, extensions: Optional[list] = None, recursive=True):
    """
    List all files in a directory.

    Args:
        path (str): The path to the directory.
        extensions (Optional[list]): A list of file extensions to filter the files. Defaults to None.
        recursive (bool): Whether to search for files recursively in subdirectories. Defaults to True.

    Returns:
        list: A list of file paths.

    """
    _base_path = os.path.join(path, "**")

    if extensions is None:
        extensions = [""]

    files = [glob(os.path.join(_base_path, f"*{ext}"), recursive=recursive) for ext in extensions]

    return list(*itertools.chain(filter(None, files)))


def insert_in_tensor(t_in: torch.tensor, value: Any):
    """
    Inserts a value into a PyTorch tensor.

    Args:
        t_in (torch.tensor): The input tensor.
        value (Any): The value to be inserted.

    Returns:
        torch.tensor: The tensor with the value inserted.

    Raises:
        ValueError: If the value is an iterable.

    """
    if isinstance(value, Iterable):
        raise ValueError("Inappropriate usage of <insert_in_tensor>. Value must not be an iterable")

    return torch.cat([t_in, torch.tensor(value).reshape(1)])


if __name__ == "__main__":
    n = 4
    d = 2
    print(get_eval_w(n, d))
