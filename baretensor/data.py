import numpy as np

from .tensor import Tensor


class Dataset:
    """Abstract base class for datasets.

    Subclasses must override :meth:`__len__` and :meth:`__getitem__`.
    """

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TensorDataset(Dataset):
    """Dataset wrapping tensors (numpy arrays or baretensor Tensors).

    Each sample is retrieved by indexing along the first dimension.
    All tensors must have the same size in dimension 0.

    Args:
        *tensors: One or more numpy arrays or baretensor Tensor objects.

    Raises:
        ValueError: If tensors differ in size along dimension 0 or are empty.
    """

    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor")

        # Convert any baretensor Tensors to numpy arrays
        converted = [
            t.data if isinstance(t, Tensor) else np.asarray(t, dtype=np.float32)
            for t in tensors
        ]

        length = converted[0].shape[0]
        for t in converted:
            if t.shape[0] != length:
                raise ValueError(
                    f"All tensors must have the same size in dimension 0, "
                    f"got {length} and {t.shape[0]}"
                )

        self.tensors = tuple(converted)
        self.length = length

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.length

    def __getitem__(self, idx):
        """Return a tuple of elements at index *idx* (one per wrapped tensor)."""
        return tuple(t[idx] for t in self.tensors)


class DataLoader:
    """Iterates over a :class:`Dataset` in mini-batches.

    Batches are yielded as tuples of numpy arrays (one per dataset tensor).
    This is *not* a multi-process loader — it performs simple, efficient
    batching in a single process.

    Args:
        dataset: A :class:`Dataset` instance.
        batch_size: Number of samples per batch (default 1).
        shuffle: If ``True``, permute indices at the start of each epoch.
        drop_last: If ``True``, drop the last incomplete batch when
            ``len(dataset)`` is not evenly divisible by ``batch_size``.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        """Return the number of batches per epoch."""
        n = len(self.dataset)
        bs = self.batch_size
        if self.drop_last:
            return n // bs
        return max(1, (n + bs - 1) // bs)  # ceiling division, but at least 1

    def __iter__(self):
        """Yield batches as tuples of numpy arrays.

        Each yielded tuple contains one numpy array per tensor in the
        underlying :class:`TensorDataset`, sliced along dimension 0
        for the indices in that batch.
        """
        n = len(self.dataset)
        bs = self.batch_size

        # Build index order
        indices = np.arange(n, dtype=np.intp)
        if self.shuffle:
            np.random.shuffle(indices)

        # Early exit for empty dataset (edge case)
        if n == 0:
            return

        # Compute the number of full batches
        num_full = n // bs

        # Yield full batches via efficient slice-based indexing
        tensors = self.dataset.tensors
        for i in range(num_full):
            start = i * bs
            end = start + bs
            batch_idx = indices[start:end]
            yield tuple(t[batch_idx] for t in tensors)

        # Handle the remainder (last incomplete batch)
        remainder = n - num_full * bs
        if remainder > 0 and not self.drop_last:
            start = num_full * bs
            batch_idx = indices[start:]
            yield tuple(t[batch_idx] for t in tensors)
