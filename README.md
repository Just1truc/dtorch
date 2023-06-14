
# DTorch

A direct link to the package : [here](https://pypi.org/project/dtorch/)

<p align="center">
<img src="https://raw.githubusercontent.com/Just1truc/dtorch/main/.asset/2023-06-07-09%2038%2034-screenshot.png" data-canonical-src="https://raw.githubusercontent.com/Just1truc/dtorch/main/.asset/2023-06-07-09%2038%2034-screenshot.png" width="200" height="200"/></p>

DTorch is a package made by a student at Epitech to improve his understanding of **pytorch** and better his knowledge of ia.
It is built on top of **numpy** which is a scientific framework for math between matrices.

This project run on cpu but still have decent computation time making it usable for model building, optimization and saving/loading.

The tensors created can work with numpy but remember that the gradients will **not be calculated** if the operation are not in the range of control of the library. Therefore, a usage of the packages methods on the tensors if preferable.

**Cuda** support may appear in the future and similarly for the dkldnn library that seems to be an excellent when working on CPU.

A direct **advantage** of using dtorch is it's lightness. The package is currently close to 14 Ko and is fast to load in any project while the use of torch often lead to a slow start.

## More about DTorch

The package is organised in the following manner :

| Module | Description |
|-|-|
| dtorch | the base module where other are regrouped|
| functionnal | a gather of all the functions that can be used on jtensors|
| jtensors | a simple module containing the JTensors declaration |
| loss | loss classes for model implementation |
| nn | a neural network layer system and its multiple layers (see doc for more details).|
| einops | a basic reproduction of some einops library methods that support autograd |
| optim | A library contaning the optimizers that can be used on the networks parameters. |
| dtorchvision | A lib containing visual datasets and models (pretrained too) |
| dtorchtext | A lib containing textual datasets and models (pretrained too) |

### About Autograd

The implementation in this library may not be the most effective computationnaly but the goal was to make it simple and make it work effectively by relying on numpy matrices operations for computation speed concerns.
