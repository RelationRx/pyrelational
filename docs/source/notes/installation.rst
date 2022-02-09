Installation
============

We do not recommend to install our library as a root user on your
system, but to set up an environment instead, using for example
`Anaconda <https://conda.io/projects/conda/en/latest/user-guide/install>`__.
To use the library, **you will need Python 3.8 or newer**.

Installation via Pip Wheels
---------------------------

You can install the pyrelational library directly using pip:

::

   pip install pyrelational

Installation from Source
------------------------

Alternatively, you can install pyrelational directly from source:

1. install the relevant packages

::

   pip install numpy>=1.20,
   pip install pandas>=1.3,
   pip install pytorch-lightning>=1.5,
   pip install torch>=1.9.0,
   pip install scikit-learn>=1.0.2,

2. install additional packages to play with our examples:

::

   pip install torchvision>=0.10.0
   pip install gpytorch>=1.4
