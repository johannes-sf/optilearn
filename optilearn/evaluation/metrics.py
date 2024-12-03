import torch


def get_binary_precision_recall_matrix(soft_labels):
    """
    Compute the binary precision and recall matrix.

    Args:
        soft_labels (torch.Tensor): The soft labels tensor.

    Returns:
        torch.Tensor: The precision-recall matrix.
    """
    epsilon = 0

    precision = get_binary_precision(soft_labels, epsilon=epsilon)
    recall = get_binary_recall(soft_labels, epsilon=epsilon)

    # Reshaping to Batch x (N_obj x N_classes)
    pr_matrix = torch.cat([precision.unsqueeze(1), recall.unsqueeze(1)], dim=1)

    return pr_matrix


def get_binary_precision(soft_labels, epsilon: float = 0.05):
    """
    Compute the binary precision.

    Args:
        soft_labels (torch.Tensor): The soft labels tensor.
        epsilon (float, optional): The epsilon value for thresholding. Defaults to 0.05.

    Returns:
        torch.Tensor: The binary precision tensor.
    """
    return soft_labels >= (1 - epsilon)


def get_binary_recall(soft_labels, epsilon: float = 0.05):
    """
    Compute the binary recall.

    Args:
        soft_labels (torch.Tensor): The soft labels tensor.
        epsilon (float, optional): The epsilon value for thresholding. Defaults to 0.05.

    Returns:
        torch.Tensor: The binary recall tensor.
    """
    return soft_labels > epsilon


def apply_criticality_preference(t_in, preferences, critical_class_mask):
    """
    Apply criticality preference to the input tensor.

    Args:
        t_in (torch.Tensor): The input tensor.
        preferences (torch.Tensor): The preferences tensor.
        critical_class_mask (torch.Tensor): The mask tensor indicating critical classes.

    Returns:
        torch.Tensor: The weighted tensor after applying criticality preferences.
    """
    critical_weights = preferences[:, 0].reshape(-1, 1)
    non_critical_weights = preferences[:, 1].reshape(-1, 1)

    weighted_t = t_in

    weighted_t[:, critical_class_mask] = weighted_t[:, critical_class_mask] * critical_weights
    weighted_t[:, ~critical_class_mask] = weighted_t[:, ~critical_class_mask] * non_critical_weights

    return weighted_t
