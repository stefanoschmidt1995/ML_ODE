# ML_ODE

``ML_ODE`` provide the code to build a Machine Learning model for solving a Ordinary differential equation (ODE).
An ODE problem consist in finding a function of time `t` ``f(t;x_0, Omega)``, dependent parametrically on `x_0` and `Omega`, such that:

```
f'(t) = F(t,x,Omega)
f(t=0) = x_0
```
in this contex `x_0` is the value of the function at some initial time and `Omega` is a set of parameters useful to specify the time derivative of the problem.

As it is well known, a neural network can approximate (at least in principle) any real function, so: why not approximating the ODE solution with a Neural Network? In doing so we follow the seminal work [Artificial Neural Networks for Solving Ordinary and Partial Differential Equations](https://arxiv.org/abs/physics/9705023). The idea is further developed in a more modern [paper](https://arxiv.org/abs/2006.14372) and this is where we take inspiration from.
Basically, we use the following ansatz for the solution:

```
f(t;x_0, Omega) = x_0 + (1-exp(-t))*NN(t, x_0, Omega)
```

This ansatz is pretty smart: it satisfies identically the boundary conditions and you can easily check how far is from the actual soluttion by comparing its time derivative with the function `F(t,Omega)` of the problem. This idea is implemented in the loss function. Once you have a loss function, it's all done: Tensorflow takes care of minimizing your loss function with respect to the weights of the NN.

## How it works

All the interesting code is wrapped in a single class, which provides you a (almost) ready to use model to fit. It is built to resemble a Keras model. You will find method `fit()` to train the model and method `call()` to evaluate the NN. You can use methods `get_solution()` or `ODE_solution()` to compute the solution that you are looking for.
You can inherit the provided baseclass and customize your own model easily. You only need to specify the following:

 - The network architecture (by writing function `set_model_properties()`), given as a list of dense keras layers (list should be `self._l_list`)
 - Some hyperparameters of the networks: the dimensionality of the problem (`n_vars`), the number of variables Omega (`n_params`), the range in which every variable lives (in list `constraints`), as well as the range of time the model should be trained at. You might also set the value of a regularizer for the loss function, you can safely use a small value, such as 0.1~1. (See lambda parameter in eq.3 [here](https://arxiv.org/abs/2006.14372) for more information)
 - The function `F(t,x,Omega)`. It should be a tensorflow function, which outputs a batch with shape `(None, n_vars)`

That's all. Not simple enough? Below you will find a simple example:

```Python
import ML_ODE
class ML_ODE(ML_ODE.ML_ODE_Basemodel):
	"Class for actually solving a ODE problem"

	def set_model_properties(self):
		"This class should initialise some important values"
		self.n_vars = 1 #number of variables of the ODE
		self.n_params = 1 #number of parameters of the ODE
			#constraints for the time integration and for each variable/parameter
			#order (t_range, x0_range, Omega1_range)
		self.constraints = [(0.,20.),(.0, 1.), (0.,1.)] 
		self.regularizer = 1. #regularizer for the loss function L'(t) = L(t) *exp(regularizer*t)

	def build_NN(self):
		"This class builds the fully connected NN architechture. Layer sequence should be provided by a list of layers (in self._l_list)"
		self._l_list.append(tf.keras.layers.Dense(128/2, activation=tf.nn.sigmoid) )
		self._l_list.append(tf.keras.layers.Dense(self.n_vars, activation=tf.keras.activations.linear))
	
	def ODE_derivative(self, z, D, Omega): #actually z, D_L, Omegas (N,2)
		"Class (in tensorflow) with the derivative of the function F(x,t). Output shape should be (None,n_vars) or (None,)"
```

You can find more examples in basic\_example.py and cosmological\_model.py.

## Installing

It's super simple: you only need the file ML_ODE.py, which keeps all the code you need. Get it with:

`wget https://raw.githubusercontent.com/stefanoschmidt1995/ML_ODE/main/ML_ODE.py`

and you're done! You can `import ML_ODE` from every python script you like: that's all. 

If you need also the examples, you might want to clone the whole repository

`git clone https://github.com/stefanoschmidt1995/ML_ODE.git`


## Documentation

Every function is documented (I did my best to make it clear).
Once you have instantiated a model, you can use the python help function for understanding the behaviour of a method

``help(ML_ODE.method_you_want_to_know_more_about)``

Furthermore, you can have a look at the examples I provided.

For more information, you can contact me at [stefanoschmidt1995@gmail.com](mailto:stefanoschmidt1995@gmail.com)
