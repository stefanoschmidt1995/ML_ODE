"""
This script uses the class ML_ODE for computing the relation between luminosity distance and redshift, as a function of the cosmological parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'..')
from ML_ODE import *

##Creating the basic class, which inherits from ML_ODE
class Cosmological_model(ML_ODE_Basemodel):
	"Class for the cosmological model."

	def set_model_properties(self):
		self.n_vars = 1	#number of variables in the problem
		self.n_params = 2 #number of parameters in the problem
		self.constraints = [(0.,20.),(.0, .0), (0.,1.), (0.,1.)] #(z_range, z0_range, Omega_M_range, Omega_L_range
		self.regularizer = .4 #regularizer for the loss function (not compulsory, default 0)

	def build_NN(self):
		self._l_list.append(tf.keras.layers.Dense(128*2, activation=tf.nn.sigmoid) )
		self._l_list.append(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid) )
		self._l_list.append(tf.keras.layers.Dense(self.n_vars, activation=tf.keras.activations.linear))
	
	def ODE_derivative(self, z, D, Omega): #actually z, D_L, Omegas (N,2)
		"""
		Derivative of the dimensionless line-of-sight comoving distance dD/dz = 1/sqrt(E(z))
		Omega[i,:] = Omega_M, Omega_Lambda
		"""
		x = 1+z
		E_z = tf.math.multiply(tf.math.pow(x, 3), Omega[:,0])
		E_z = E_z + tf.math.multiply(tf.math.pow(x, 2), 1-Omega[:,0]-Omega[:,1]) + Omega[:,1]  #(N,)
		return tf.math.reciprocal(tf.math.sqrt(E_z))

	def luminosity_distance(self,t, X_0, Omega, h = 0.7):
		"""
		Computes the luminosity distance (in Mpc), given redshift and the Omegas. (Wrapper to ODE_solution, specific to the cosmological model)
		See eq. 16 and 21 in https://arxiv.org/abs/astro-ph/9905116
		"""
		const = 3000./h #c /H_0 (in Mpc)
		Omega_k = 1. - Omega[0] - Omega[1]
		D_c = self.ODE_solution(t, X_0, Omega) #(1,)
		if Omega_k > 1e-7:
			D_c = np.sinh(np.sqrt(Omega_k)*D_c)/np.sqrt(Omega_k)
		if Omega_k < 1e-7:
			D_c = np.sin(np.sqrt(np.abs(Omega_k))*D_c)/np.sqrt(np.abs(Omega_k))
		return const * D_c


def plot(model, savefile, show = False):
	plot_solution(model, 10, [0.], seed = 0, folder = savefile, show = show)


#Building and fitting the model
what_to_do = "load"

model_name = "cosmo_model"
model = Cosmological_model(model_name)
model_file = "{}/{}".format(model_name, model_name) #file which holds the model

	#deciding what to do
if what_to_do == 'load':
	model.load_everything(model_file)
elif what_to_do == 'fitload' or what_to_do == 'fit':
	if what_to_do == 'fitload':
		model.load_everything(model_file)
	model.fit(20000, N_batch = 20000,  learning_rate = 5e-4, save_output = True, plot_function = plot, save_step = 10, print_step = 10)
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
plt.savefig("images/loss_cosmo.png", transparent =False)

plot(model, None, True)










