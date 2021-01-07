# Examples of the usage of ML_ODE

Here you can find two examples of a working (and fitted model).
The first one `example_model.py` is a dummy (but non-trivial) example of solving an equation.
The second one `cosmological_model.py` is inspired by cosmology: it learns the relation between luminosity distance and redshift as a function of the content of the Universe (parametrized by Omega).

You can find the fitted models in the respective folders. Each of them hols the final model, with the loss function and the weights (loadable with function `load_everything()`).
The subfolders keep track of the model during the training: each of them, labeled with the number of epoch it refers to, holds a plot with a comparison between NN's output and true ODE solution and the weights of the partially trained model.

Enjoy! :)
