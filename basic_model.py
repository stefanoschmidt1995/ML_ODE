"""
This script is intented to present a basic example of how to use the class ML_ODE
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ML_ODE import *

##Creating the basic class, which inherits from ML_ODE
class Basic_example(ML_ODE_Basemodel):
	"""
	Basic example on how to use ML_ODE. This class inherrits from ML_ODE_Basemodel and implements a few methods. Then everything is ready.
	It solves the linear two dimensional problem:
		x'(t) = A x(t)
		x(0) = x_0
	"""

	def set_model_properties(self):
		"This class should initialise some important values for the problem"
		self.n_vars = 2	#number of variables in the problem
		self.n_params = 4 #number of parameters in the problem
		self.constraints = [(0.,1.),(-1., 1.), (-1., 1.), (-1., 1.),(-1., 1.),(-1., 1.),(-1., 1.)] #(t_range, x0_range, y0_range, Omega_i_range)
		self.regularizer = 1. #regularizer for the loss function (not compulsory, default 0)

	def build_NN(self):
		"Here the architechture of the network is given. The list self._l_list should be filled with keras layers"
		self._l_list.append(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid) )
		self._l_list.append(tf.keras.layers.Dense(128/2, activation=tf.nn.sigmoid) )
		self._l_list.append(tf.keras.layers.Dense(self.n_vars, activation=tf.keras.activations.linear))
	
	def ODE_derivative(self, t, x, Omega): #actually z, D_L, Omegas (N,2)
		"""
		Here we write down the derivative of x; in this case A * x.
		Inputs:
			t (None,)					times
			x (None, n_vars)			values of the variables
			Omega (None, n_params)		parameters
		Output shape should be (None,n_vars) (or also (None,) if n_vars is 1)
		"""
		A = tf.reshape(Omega, [tf.shape(Omega)[0], self.n_vars,self.n_vars]) #(N_batch, 2,2)
		x = tf.expand_dims(x, axis=-1) #(N_batch, 2,1)
		res = tf.matmul(A,x) #(N_batch, 2,1)
		return tf.squeeze(res, axis = -1) #(N_batch, 2)


def plot(model, savefile, show = False):
	plot_solution(model, 10, [-0.5, .5], seed = 0, folder = savefile, show = show)


#Building and fitting the model
what_to_do = "fit"

model_name = "example_model"
model = Basic_example(model_name)
model_file = "{}/{}".format(model_name, model_name) #file which holds the model

	#deciding what to do
if what_to_do == 'load':
	model.load_everything(model_file)
elif what_to_do == 'fitload' or what_to_do == 'fit':
	if what_to_do == 'fitload':
		model.load_everything(model_file)
	model.fit(2000000, N_batch = 20000,  learning_rate = 5e-4, save_output = True, plot_function = plot, save_step = 5000, print_step = 100)
	model.save_weights(model_file)
elif what_to_do == 'fit':
	pass
else:
	print("Didn't understand what you asked: sorry :(")
	quit()


history = np.array(model.history)
metric = np.array(model.metric)
plt.figure()
plt.plot(history[:,0], history[:,1], c = 'b')
plt.plot(metric[:,0], metric[:,1], c = 'r')
plt.ylabel('loss/metric')
plt.yscale('log')
plt.savefig(model_name+"/loss.pdf", transparent =True)

plot(model, ".", True)









