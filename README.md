# ML_ODE

``ML_ODE`` provide the code to build a Machine Learning model for solving a Ordinary differential equation (ODE).

## How it works

All the interesting code is wrapped in a single class, which resembles a Keras model. It support methods fit and call and it is a compact way of encoding a model.
You can inherit the provided baseclass and build their own model easily.
Below a simple example:

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

You can find more examples in basic\_example.py cosmological\_model.py.

## Installing

It's super simple:

``git clone https://github.com/stefanoschmidt1995/ML_ODE.git``

and you're done! You can import ML_ODE from every python script you like: that's all. 

## Documentation

Every function is documented (I did my best to make it a clear documentation).
Once you have instantiated a model, you can use the python help functions

``help(ML_ODE.function)``

For more information, you can contact me at [stefanoschmidt1995@gmail.com](mailto:stefanoschmidt1995@gmail.com)



