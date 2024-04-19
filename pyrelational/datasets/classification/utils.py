from torch import Tensor


def remap_to_int(torch_class_array: Tensor) -> Tensor:
    """
    Remap the elements in a torch tensor to contiguous integers starting from 0,
    which is useful for classification tasks where class labels should start from zero and be contiguous.

    :param torch_class_array: A torch.Tensor containing class labels, possibly non-integer or non-contiguous.
    :return: A torch.Tensor with class labels remapped to integers starting from 0.

    Example:
        >>> torch_class_array = torch.tensor([10, 10, 20, 20, 30])
        >>> remap_to_int(torch_class_array)
        tensor([0, 0, 1, 1, 2])
    """
    remapped_labels: Tensor = torch_class_array.unique(return_inverse=True)[1]
    return remapped_labels
