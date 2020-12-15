"""
This script illustrates baisc usage for the class ML_ODE by computing the relation between luminosity distance and redshift, as a function of the cosmological parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from ML_ODE import *

##Creating the basic class, which inherits from ML_ODE

class Cosmological_model(ML_ODE_Basemodel):
	"Class for the cosmological model."

	def set_model_properties(self):
		self.n_vars = 1
		self.n_params = 1
		self.ranges = [(-.1,1.), (0.,1.), (0.,1.)]
		self.regularizer = 1. #regularizer for the loss function (not compulsory, default 0)

	def build_NN(self):
		self._l_list.append(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid) )
		self._l_list.append(tf.keras.layers.Dense(128/2, activation=tf.nn.sigmoid) )
		self._l_list.append(tf.keras.layers.Dense(1, activation=tf.keras.activations.linear))
	
	def ODE_derivative(self, t, X, Omega): #actually z, D_L, Omegas
		return tf.math.multiply(-Omega, X)



def plot(model, savefile, show = False):
	plot_solution(model, 10, [1.],seed = 0, folder = savefile, show = show)


#Building and fitting the model
what_to_do = "load"

model_name = "cosmo_model_try"
model = Cosmological_model(model_name)
model_file = "{}/{}".format(model_name, model_name) #file which holds the model


if what_to_do == 'load':
	model.load_everything(model_file)
elif what_to_do == 'fitload':
	model.load_everything(model_file)
	what_to_do = 'fit'
elif what_to_do == 'fit':
	model.fit(20000, N_batch = 20000,  learning_rate = 5e-4, save_output = True, plot_function = plot, save_step = 10, print_step = 10)
	model.save_weights(model_file)
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









