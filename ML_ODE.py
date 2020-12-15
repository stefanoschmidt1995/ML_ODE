"""
Basic class for fitting a Neural Network which solves a ODE.
"""

try:
	import silence_tensorflow.auto #awsome!!!! :)
except:
	pass

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

#defining a Model
class ML_ODE_Basemodel(tf.keras.Model):

	def __init__(self, name = "ML_ODE_model"):
		super(ML_ODE_Basemodel, self).__init__(name = name)
		print("Initializing model ",self.name)
		self.history = []
		self.metric = []
		self.epoch = 0
		self.regularizer = 0.

			#initializing ranges and checking
		self.set_model_properties()
		if not isinstance(self.ranges,list):
			raise ValueError("List ranges should be of list type with format [(min_t, max_t), (min_var1, max_var1), ..., (max_par1, max_parD)]")
		for _range in self.ranges:
			if not isinstance(_range,list) and not isinstance(_range,tuple):
				raise ValueError("Each element in range list should be a tuple or a list")
			if not isinstance(_range[0],float) or not isinstance(_range[1],float):
				raise ValueError("Limits for each variable should be a float")

			#setting a positive regularizer (just in case)
		self.regularizer = np.abs(self.regularizer)

		self._l_list = []
		self.build_NN() #implemented in child class

		for l in self._l_list:
			if not isinstance(l, tf.keras.layers.Dense):
				raise ValueError("Mehod build_NN should provide the NN architecture as a list of tf.keras.layers.Dense objects.")

		self.optimizer = tf.keras.optimizers.Adam(learning_rate = 5e-4) #default optimizer
		self.build(input_shape = (None, 1+self.n_vars+self.n_params)) #This is required to specify the input shape of the model and to state which are the trainable paramters

	def build_NN(self):
		raise NotImplementedError("Function build_NN should be implemented in the child class.")

	def set_model_properties(self):
		raise NotImplementedError("Function set_ODE_properties should be implemented in the child class.")

	def ODE_derivative(self, t, X, Omega):
		raise NotImplementedError("Function ODE_derivative should be implemented in the child class")

	def call(self, inputs):
		"Inputs: [t, X_0 (n_vars,), Omega (n_params,)]"
		output = inputs
		for l in self._l_list:
			output = l(output)
		return output #(N,n_vars)

	def get_solution(self, inputs):
		"Inputs are (N,7)"
		return inputs[:,1:self.n_params+1] + tf.transpose(tf.math.multiply( tf.transpose(self.call(inputs)), 1-tf.math.exp(-self.regularizer*inputs[:,0])) ) #(N,n_vars)

	def __ok_inputs(self, inputs):
		if not isinstance(inputs, tf.Tensor):
			inputs = tf.convert_to_tensor(inputs, dtype=tf.float32) #(N,D)
			if inputs.ndim == 1:
				inputs = inputs[None,:]
		return inputs

	def ODE_derivative_np(self,t, X, Omega):
		return self.ODE_derivative(tf.convert_to_tensor(t,dtype = tf.float32), tf.convert_to_tensor(X,dtype = tf.float32), tf.convert_to_tensor(Omega,dtype = tf.float32)).numpy()

	def ODE_solution(self,t, X_0, Omega):
		"Numpy interface for the solution of the ODE with ML. Accepts a list of times, the initial conditions (n_vars,) and Omega (n_params,)."
		X_0 = np.array(X_0)
		Omega = np.array(Omega)
		assert Omega.shape == (self.n_vars,)
		assert X_0.shape == (self.n_params,)
		X = np.repeat([[*X_0, *Omega]],len(t), axis = 0) #(T,3)
		X = np.concatenate([np.array(t)[:, None],X], axis = 1) #(T,1+model.n_vars)
		X = self.__ok_inputs(X) #casting to tf
		res = self.get_solution(X) #(T,3)
		return res.numpy()
	
	def loss(self, X):
		"""
		Loss function: takes an array X (N,1+3+3) with values to test the model at. X[0,:] = [t, (x0)_0, (x0)_1, (x0)_2, (Omega)_0, (Omega)_1, (Omega)_2]
		Input should be tensorflow only.
		"""
		Omega = X[:,self.n_vars+1:]
		with tf.GradientTape() as g:
			g.watch(X)
			out = self.get_solution(X)
		
		grad = g.batch_jacobian(out, X)[:,:,0] #d/dt #(N,3)
		F = self.ODE_derivative(X[:,0], out, Omega)

			#loss can be multiplied by exp(-alpha*t) for "regularization"
		loss = tf.math.square(grad - F) #(N,3)
		loss = tf.transpose(tf.math.multiply(tf.transpose(loss), tf.math.exp(-1.*X[:,0]))) #(N,3)
		loss = tf.reduce_sum(loss, axis = 1) /X.shape[1] #(N,)
		return loss

	@tf.function#(jit_compile=True) #very useful for speed up
	def grad_update(self, X):
		"Input should be tensorflow only."
		with tf.GradientTape() as g:
			g.watch(self.trainable_weights)
			loss = tf.reduce_sum(self.loss(X))/X.shape[0]

		gradients = g.gradient(loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

		return loss

	def get_random_X(self, N_batch, seed = None):
		if isinstance(seed,int):
			tf.random.set_seed(seed)
		else:
			tf.random.set_seed(np.random.randint(0,1000000))

		random_Xs = []
		for _range in self.ranges:
			random_Xs.append(tf.random.uniform((N_batch,1), minval=_range[0], maxval=_range[1], dtype=tf.dtypes.float32))

		return tf.concat(random_Xs, axis = 1) #(N_batch, 7)
	
	def fit(self, N_epochs, N_batch = 20000,  learning_rate = 5e-4, save_output = True, plot_function = None, save_step = 20000, print_step = 10):
		self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate) #default optimizer
		epoch_0 = self.epoch
		for i in range(N_epochs):
			X = self.get_random_X(N_batch)

			loss = self.grad_update(X)

			if i % print_step == 0: #saving history
				self.epoch = epoch_0 + i
				self.history.append((self.epoch, loss.numpy()))
				print(self.epoch, loss.numpy())
				if save_output:
					self.save_weights("{}/{}".format(self.name, self.name)) #overwriting the newest
					np.savetxt(self.name+"/"+self.name+".loss", np.array(self.history))
					np.savetxt(self.name+"/"+self.name+".metric", np.array(self.metric))

			if i == 0: continue

			if save_output:
				if i%save_step ==0: #computing metric loss
					metric = 0.
					N_avg = 100 #batch size to compute the metric at
					X = self.get_random_X(N_avg)
					times = np.linspace(0.,10.,100)
					for j in range(N_avg):
							#solving ODE for solution
						X_t = scipy.integrate.odeint(self.ODE_derivative_np, X[j,1:self.n_vars+1].numpy(), times, args = (X[j,self.n_vars+1:],), tfirst = True)

						X_t_NN = self.ODE_solution(times, X[j, 1:self.n_vars+1], X[j, self.n_vars+1:]) #(D,)
						plt.plot(times, X_t_NN)

						metric += np.mean(np.square(X_t -X_t_NN))

					self.metric.append((self.epoch, metric/N_avg))
					print("\tMetric: {} {}".format(self.metric[-1][0],self.metric[-1][1]))

					self.save_weights("{}/{}/{}".format(self.name, str(self.epoch), self.name)) #saving to arxiv
					if plot_function is not None:
						plot_function(self, "{}/{}".format(self.name, str(self.epoch)))
						
					
		return self.history

	def load_everything(self, path):
		"Loads model and tries to read metric and loss"
		print("Loading model from: ",path)
		self.load_weights(path)
		try:
			self.history = np.loadtxt(path+".loss").tolist()
			self.epoch = int(self.history[-1][0])
		except:
			self.epoch = 0
			pass

		try:
			self.metric = np.loadtxt(path+".metric").tolist()
		except:
			pass

		return


####### Helpers ##############

def plot_solution(model, N_sol, X_0,  seed, folder = ".", show = False):
	X = model.get_random_X(N_sol, seed).numpy()
	X[:,1:1+model.n_vars] = X_0

	times = np.linspace(0.,10.,200)
	X_t = np.zeros((N_sol, times.shape[0],3))
	X_t_rec = np.zeros((N_sol, times.shape[0],3))


	for i in range(N_sol):
		X_t_rec[i,:,:] = model.ODE_solution(times, X[i,1:1+model.n_vars], X[i,1+model.n_vars:])
		X_t[i,:,:] = scipy.integrate.odeint(model.ODE_derivative_np, np.array(X[i,1:1+model.n_vars]), times, args = (np.array(X[i,1+model.n_vars:]),), tfirst = True)

	for var in range(model.n_vars):

		plt.figure()
		for i in range(N_sol):
			true, = plt.plot(times,X_t[i,:,1+var],  c = 'r')
			NN, = plt.plot(times,X_t_rec[i,:,1+var], c = 'b')
		plt.xlabel(r"$t$")
		plt.ylabel(r"$L_x$")
		plt.legend([true, (true, NN)], ["True", "NN"])


		plt.savefig(folder+"/var{}.pdf".format(var), transparent =True)


	if show:
		plt.show()
	else:
		plt.close('all')





