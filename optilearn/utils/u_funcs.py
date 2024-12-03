from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
from torch import nn

EPSILON = 1e-6


class AbstractUFunc(ABC):
    """
    Abstract base class for utility functions (u_funcs).
    """

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, r, pref, dim=-1, numpy=False):
        """
        Calculate the utility value based on the input.

        Args:
            r (torch.Tensor or np.ndarray): Input tensor or array.
            pref (torch.Tensor or np.ndarray): Preference tensor or array.
            dim (int): Dimension along which to calculate the utility value. Default is -1.
            numpy (bool): Whether to use numpy or torch for calculations. Default is False.

        Returns:
            torch.Tensor or np.ndarray: The calculated utility value.
        """
        pass

    @staticmethod
    def _sign_log_abs(x, numpy=False):
        """
        Calculate the sign * log(abs(x) + epsilon) of the input.

        Args:
            x (torch.Tensor or np.ndarray): Input tensor or array.
            numpy (bool): Whether to use numpy or torch for calculations. Default is False.

        Returns:
            torch.Tensor or np.ndarray: The calculated value.
        """
        if numpy:
            return np.sign(x) * np.log(np.abs(x) + EPSILON)
        return torch.sign(x) * torch.log(torch.abs(x) + EPSILON)


class Linear(AbstractUFunc):
    """
    Linear utility function class.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, r, pref, dim=-1, numpy=False):
        """
        Calculate the utility value using the linear utility function.

        Args:
            r (torch.Tensor or np.ndarray): Input tensor or array.
            pref (torch.Tensor or np.ndarray): Preference tensor or array.
            dim (int): Dimension along which to calculate the utility value. Default is -1.
            numpy (bool): Whether to use numpy or torch for calculations. Default is False.

        Returns:
            torch.Tensor or np.ndarray: The calculated utility value.
        """
        if numpy:
            return (r * pref).sum(axis=dim)
        return (r * pref).sum(dim=dim)


class Square(AbstractUFunc):
    """
    Square utility function class.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, r, pref, dim=-1, numpy=False):
        """
        Calculate the utility value using the square utility function.

        Args:
            r (torch.Tensor or np.ndarray): Input tensor or array.
            pref (torch.Tensor or np.ndarray): Preference tensor or array.
            dim (int): Dimension along which to calculate the utility value. Default is -1.
            numpy (bool): Whether to use numpy or torch for calculations. Default is False.

        Returns:
            torch.Tensor or np.ndarray: The calculated utility value.
        """
        if numpy:
            return (r * pref).sum(axis=dim)
        return (self._sign_pow_abs(r, 2, numpy=numpy) * pref).sum(axis=dim)

    @staticmethod
    def _sign_pow_abs(x, p, numpy=False):
        """
        Calculate the sign * pow(abs(x), p) of the input.

        Args:
            x (torch.Tensor or np.ndarray): Input tensor or array.
            p (int): Power value.
            numpy (bool): Whether to use numpy or torch for calculations. Default is False.

        Returns:
            torch.Tensor or np.ndarray: The calculated value.
        """
        if numpy:
            return np.sign(x) * np.exp(np.abs(x), p)
        return torch.sign(x) * torch.pow(torch.abs(x), p)


class Log(AbstractUFunc):
    """
    Log utility function class.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, r, pref, dim=-1, numpy=False):
        """
        Calculate the utility value using the log utility function.

        Args:
            r (torch.Tensor or np.ndarray): Input tensor or array.
            pref (torch.Tensor or np.ndarray): Preference tensor or array.
            dim (int): Dimension along which to calculate the utility value. Default is -1.
            numpy (bool): Whether to use numpy or torch for calculations. Default is False.

        Returns:
            torch.Tensor or np.ndarray: The calculated utility value.
        """
        r = r + 1
        r = r - (r < 1) * 2
        return (self._sign_log_abs(r, numpy=numpy) * pref).sum(axis=dim)

    @staticmethod
    def _sign_log_abs(x, numpy=False):
        """
        Calculate the sign * log(abs(x) + epsilon) of the input.

        Args:
            x (torch.Tensor or np.ndarray): Input tensor or array.
            numpy (bool): Whether to use numpy or torch for calculations. Default is False.

        Returns:
            torch.Tensor or np.ndarray: The calculated value.
        """
        if numpy:
            return np.sign(x) * np.log(np.abs(x) + EPSILON)
        return torch.sign(x) * torch.log(torch.abs(x) + EPSILON)
