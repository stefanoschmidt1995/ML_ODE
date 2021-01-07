# Examples of the usage of ML_ODE

Here you can find two examples of a working (and fitted model).

`example_model.py` is a dummy (but non-trivial) example of solving an equation.
It solves the equation (2 dimensions): `x' = A x`, where A is an arbitrary 2x2 matrix.
Here are the results:



First variable             |  Second variable
:-------------------------:|:-------------------------:
[<object data="https://github.com/stefanoschmidt1995/ML_ODE/raw/main/examples/example_model/380000/var0.pdf" type="application/pdf" width="700px" height="700px">    <embed src="https://github.com/stefanoschmidt1995/ML_ODE/raw/main/examples/example_model/380000/var0.pdf">  <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/stefanoschmidt1995/ML_ODE/raw/main/examples/example_model/380000/var0.pdf">Download PDF</a>.</p>    </embed> </object>  ]|  [<object data="https://github.com/stefanoschmidt1995/ML_ODE/raw/main/examples/example_model/380000/var1.pdf" type="application/pdf" width="700px" height="700px">    <embed src="https://github.com/stefanoschmidt1995/ML_ODE/raw/main/examples/example_model/380000/var1.pdf">  <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/stefanoschmidt1995/ML_ODE/raw/main/examples/example_model/380000/var1.pdf">Download PDF</a>.</p>    </embed> </object>  ]

The second one `cosmological_model.py` is inspired by cosmology: it learns the relation between luminosity distance and redshift as a function of the content of the Universe (parametrized by Omega).

You can find the fitted models in the respective folders. Each of them hols the final model, with the loss function and the weights (loadable with function `load_everything()`).
The subfolders keep track of the model during the training: each of them, labeled with the number of epoch it refers to, holds a plot with a comparison between NN's output and true ODE solution and the weights of the partially trained model.

Enjoy! :)
