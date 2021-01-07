# Examples of the usage of ML_ODE

Here you can find two examples of a working (and fitted model).

`example_model.py` is a dummy (but non-trivial) example of solving an equation.
It solves the equation (2 dimensions): `x' = A x`, where A is an arbitrary 2x2 matrix.
Here are the results:

First variable             |  Second variable
:-------------------------:|:-------------------------:
![](https://github.com/stefanoschmidt1995/ML_ODE/raw/main/examples/images/var0_example.png)| ![](https://github.com/stefanoschmidt1995/ML_ODE/raw/main/examples/images/var1_example.png)

T`cosmological_model.py` is useful in cosmology: it learns the relation between luminosity distance and redshift as a function of the content of the Universe (parametrized by Omega). More precisely, it learns the relation between redshift and dimensionless line-of-sight comoving distance (see [here](https://arxiv.org/abs/astro-ph/9905116) for more details of the conventions used).
The model works quite well up to redshift `z = 20`

![](https://github.com/stefanoschmidt1995/ML_ODE/raw/main/examples/images/var0_example.png =500x)

You can find the fitted models in the respective folders. Each of them hols the final model, with the loss function and the weights (loadable with function `load_everything()`).
The subfolders keep track of the model during the training: each of them, labeled with the number of epoch it refers to, holds a plot with a comparison between NN's output and true ODE solution and the weights of the partially trained model.

Enjoy! :)
