"Normalization functions"

import torch


def max_over_dims(tensor, dims):
    """Computes the max over specified dimensions.

    Args:
      tensor: The input tensor.
      dims: A tuple of dimensions to reduce.

    Returns:
      The reduced tensor.
    """
    return op_over_dims(torch.max, tensor, dims)


def min_over_dims(tensor, dims):
    """Computes the min over specified dimensions.

    Args:
      tensor: The input tensor.
      dims: A tuple of dimensions to reduce.

    Returns:
      The reduced tensor.
    """
    return op_over_dims(torch.min, tensor, dims)


def op_over_dims(op, tensor, dims):
    """Computes a op over specified dimensions.

    Args:
      op: The operation
      tensor: The input tensor.
      dims: A tuple of dimensions to reduce.

    Returns:
      The reduced tensor.
    """
    for dim in reversed(dims):
        tensor, _ = op(tensor, dim=dim, keepdim=True)
    return tensor.squeeze()


def normalize_batch_zero_one(batch):
    """Normalizes a batch into [0,1] interval. Normalizes by sample.

    Args:
      batch: The input tensor. Batch dim should be the first one.

    Returns:
      The normalized tensor.
    """
    view_shape = [-1]
    view_shape.extend([1] * (batch.dim() - 1))
    batch_max = max_over_dims(batch, tuple(range(batch.dim()))[1:]).view(
        view_shape
    )
    batch_min = min_over_dims(batch, tuple(range(batch.dim()))[1:]).view(
        view_shape
    )

    normalized_batch = (batch - batch_min) / (batch_max - batch_min)
    return normalized_batch
