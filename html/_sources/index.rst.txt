.. dtorch documentation master file, created by
   sphinx-quickstart on Sun Jun 18 01:54:17 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dtorch's documentation!
==================================

.. raw:: html

   <hr/>

Description
"""""""""""

DTorch is a package made by a student at Epitech to improve his understanding of **pytorch** and better his knowledge of ia.

It is built on top of **numpy** which is a scientific framework for math between matrices.

This project run on cpu but still have decent computation time making it usable for model building, optimization and saving/loading.
The tensors created can work with numpy but remember that the gradients will **not be calculated** if the operation are not in the range of control of the library. Therefore, a usage of the packages methods on the tensors if preferable.
**Cuda** support may appear in the future and similarly for the mkldnn library that seems to be excellent when working on CPU.

.. tip::
   A direct **advantage** of using dtorch is it's lightness. The package is currently close to 14 Ko and is fast to load in any project while the use of torch often lead to a slow start.

Content
"""""""

.. toctree::
   dtorch
   dtorchtree
   dtorchvision
   dtorchtext
