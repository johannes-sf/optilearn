import warnings
from typing import Union

import numpy as np
import torch
from torch.nn import functional as torch_functional

from optilearn.configs.constants import LabelType
from optilearn.environments.responses import LearningImageLoaderResponse


def transform_labels(
    sample: LearningImageLoaderResponse,
    label_type: Union[LabelType, str] = LabelType.NORMAL,
    num_classes: int = None,
):
    """

    Args:
        sample:
        label_type:
        num_classes:

    Returns:

    """
    if sample.state_one_hot_labels is None:
        sample.state_one_hot_labels = one_hot_encode_class_indices(sample.state_labels, num_classes=num_classes)

    if sample.contaminant_indices is not None:
        sample.state_soft_labels = get_soft_labels(sample)

    if isinstance(label_type, str):
        label_type = LabelType(label_type)

    if label_type == LabelType.SOFT:
        if sample.state_soft_labels is None:
            warnings.warn(
                f"Soft labels are only possible with contamination. Either set the "
                f"<contamination> value or set the <label_type> to 'one_hot' or 'normal' in env config."
            )

        sample.state_labels = sample.state_soft_labels
    elif label_type == LabelType.ONE_HOT:
        sample.state_labels = sample.state_one_hot_labels
    else:  # [0,2,4,0,1]# [[1,0,0,0,0],[0,0,1,0,0],[0,0,0,0,1],[1,0,0,0,0],[0,1,0,0,0]]
        pass

    return sample


def get_soft_labels(sample: LearningImageLoaderResponse):
    """
    Generates soft labels for a sample based on the mixture ratio
    Parameters
    ----------
    sample

    Returns
    -------

    """
    soft_labels = sample.state_one_hot_labels.float()
    mixture_mask = sample.state_labels != sample.contaminant_indices

    # For samples which aren't mixed, the contaminant index is the same as class index
    soft_labels[mixture_mask, sample.state_labels[mixture_mask]] = sample.mixture_ratios[mixture_mask]
    soft_labels[mixture_mask, sample.contaminant_indices[mixture_mask]] = 1 - sample.mixture_ratios[mixture_mask]

    return soft_labels


def one_hot_encode_labels(labels: Union[torch.Tensor, list, np.array], label_to_class_index: dict):
    """
    Turns raw class labels to one-hot-encoded vectors.

    Args:
        labels: collection of labels
        label_to_class_index: mapping of labels to corresponding class index

    Returns:

    """
    indices = torch.tensor(
        [label_to_class_index[str(label.item())] for label in labels],
        device=labels.device,
    )

    return torch_functional.one_hot(indices, num_classes=len(label_to_class_index))


def one_hot_encode_class_indices(class_indices: torch.Tensor, num_classes: int):
    """
    Turns individual class indices to one-hot-encoded vectors

    Args:
        class_indices: collection of class indices
        num_classes: total number of classes

    Returns:

    """
    return torch_functional.one_hot(class_indices, num_classes=num_classes)
