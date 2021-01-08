# Examples of the usage of ML_ODE

Here you can find two examples of a working (and fitted model). Both examples are commented and, hopefully, they provide everything that you need to build and train your own model.

You can find the fitted models in the respective folders. Each of them hols the final model, with the loss function, the metric (average mse error) and the weights (loadable with function `load_everything()`).
The subfolders keep track of the model during the training: each of them, labeled with the number of epoch it refers to, holds a plot with a comparison between NN's output and true ODE solution and the weights of the partially trained model.

Enjoy! :)

## `example_model.py`
`example_model.py` is a dummy (but non-trivial) example of solving an equation.
It solves the equation (2 dimensions): `x' = A x`, where A is an arbitrary 2x2 matrix.
Here are the results:

First variable             |  Second variable
:-------------------------:|:-------------------------:
![](https://github.com/stefanoschmidt1995/ML_ODE/raw/main/examples/images/var0_example.png)| ![](https://github.com/stefanoschmidt1995/ML_ODE/raw/main/examples/images/var1_example.png)

As you see the blue lines almost perfectly overlaps the red line: the network is well able to reproduce the solution to the ODE obtaine with standard Runge-Kutta integration.

## `cosmological_model.py`
`cosmological_model.py` is a useful cosmology example: it learns the relation between luminosity distance and redshift as a function of the content of the Universe (parametrized by Omega). More precisely, it learns the relation between redshift and dimensionless line-of-sight comoving distance (see [here](https://arxiv.org/abs/astro-ph/9905116) for more details of the conventions used).
The class implements also a function `luminosity_distance()` to compute the actual luminosity distance (eq. (16) and (21) of the paper above); of course, this function is just a helper and it is not required to fit the NN or to build the model.
The model works quite well up to redshift `z = 20`

<center>
<img src="https://github.com/stefanoschmidt1995/ML_ODE/raw/main/examples/images/var0_cosmo.png" height="500">
</center>

Again, you see the the match between red and blue is satisfying.

If you are curious, below you can see the progress in the training: the loss function (blue) and the metric (red, the average mse between the true and the reconstructed solution) are reported as a function of the epoch number

<center>
<img src="https://github.com/stefanoschmidt1995/ML_ODE/raw/main/examples/images/loss_cosmo.png" height="500">
</center>


























