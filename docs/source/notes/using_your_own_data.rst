.. _using own data:

Using your own datasets with pyrelational
==================================

The :py:class:`pyrelational.data.data_manager.GenericDataManager` module enables users to integrate any pytorch Dataset
into pyrelational easily. The module expects the full dataset, i.e. the union of labelled, unlabelled,
validation (optional), and test sets. The indices of each sets should be provided to the class constructor that
then proceeds to construct the subset Datasets object under the hood. Throughout the experiment, the data manager will
keep track of indices and handle updates to the labelled/unlabelled pools of samples. For instance, using the Mnist dataset

.. code-block:: python

   import torch
   from torchvision import datasets, transforms
   from pyrelational.data.data_manager import GenericDataManager

    mnist_dataset = datasets.MNIST(
        "mnist_data",
        download=True,
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )
    train_ds, val_ds, test_ds = torch.utils.data.random_split(mnist_dataset, [50000, 5000, 5000])
    train_indices = train_ds.indices
    validation_indices = val_ds.indices
    test_indices = test_ds.indices
    labelled_indices = train_indices[:10000]

    data_manager = GenericDataManager(
        mnist_dataset,
        train_indices=train_indices,
        labelled_indices=labelled_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )

Customizing dataloader
______________________

Users can customize the dataloaders in the same way as any pytorch dataloader by passing Pytorch DataLoader arguments to
the data manager constructor, such as

.. code-block:: python
    :emphasize-lines: 7,8,9

    data_manager = GenericDataManager(
        mnist_dataset,
        train_indices=train_indices,
        labelled_indices=labelled_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        loader_batch_size=10000,
        loader_num_workers=2,
        loader_shuffle=True,
    )

Interacting with non-pytorch estimators
_______________________________________

Importantly, this enables using pytorch Dataset and DataLoaders to interact with other libraries by taking advantage of
the collate function. For instance, using the following collate function enables conversion to numpy array

.. code-block:: python
    :emphasize-lines: 15

    def numpy_collate(batch):
        """Collate function for a Pytorch to Numpy DataLoader"""
        batchx = [item[0] for item in batch]
        batchy = [item[1] for item in batch]
        np_batchx = [x.cpu().detach().numpy() for x in batchx]
        np_batchy = [y.cpu().detach().numpy() for y in batchy]
        return [np.array(np_batchx), np.array(np_batchy)]

    data_manager = GenericDataManager(
        mnist_dataset,
        train_indices=train_indices,
        labelled_indices=labelled_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        loader_collate_fn=numpy_collate,
    )


Returning single batch
___________________________

In some instances, for instance when using Gaussian Processes or scikit-learn estimators, the dataloader should return the
entire underlying dataset. This can be specified as such,

.. code-block:: python
    :emphasize-lines: 7

    data_manager = GenericDataManager(
        mnist_dataset,
        train_indices=train_indices,
        labelled_indices=labelled_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        loader_batch_size="full",
    )
