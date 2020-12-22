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
	"""
class ML_ODE_Basemodel
======================
	Basic class for a ML model for solving a ODE:
		dx/dt = F(t,x; Omega)
		x(t=0) = x_0
	The model provides a solution in the form:
		x(t;x_0,Omega) = x_0 + (1-exp(-t)) * NN(t,x_0,Omega)
	where NN(t,x_0,Omega) is a neural network to be trained.
	See https://arxiv.org/abs/2006.14372 or https://arxiv.org/abs/physics/9705023 for more information.
	Every model should inherit from this class and implement functions set_model_properties, build_NN and ODE_derivative, as follows:
		class ML_ODE(ML_ODE_Basemodel):
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
		
		A model can be saved with load method and can be loaded either with load_weights either with load_everything.
		The model can be fitted by gradient descent with method fit(); during the training procedure the loss function is monitored and checkpoints saves are made. A mse metric function is also evaluated.
		Once the model is trained, the result can be evaluated with get_solution() (only with tf inputs/outputs). You can also use ODE_solution() for an easy to use numpy interface.
		The model must have a name, given at initialization, that is used for saving a loading operations.
	"""

	def __init__(self, name = "ML_ODE_model"):
		"""
		Initialise the model by building the NN and setting a number of parameters (i.e. #params, #variable, constraints, regularizer).
		Input:
			name (str)		a name for the model
		"""
		super(ML_ODE_Basemodel, self).__init__(name = name)
		print("Initializing model ",self.name)
		self.history = []
		self.metric = []
		self.epoch = 0
		self.regularizer = 0.

			#initializing constraints and checking
		self.set_model_properties()
		if not isinstance(self.constraints,list):
			raise ValueError("List of constraints should be of list type with format [(min_t, max_t), (min_var1, max_var1), ..., (max_par1, max_parD)]")
		for _cons in self.constraints:
			if not isinstance(_cons,list) and not isinstance(_cons,tuple):
				raise ValueError("Each element in range list should be a tuple or a list")
			if not isinstance(_cons[0],float) or not isinstance(_cons[1],float):
				raise ValueError("Limits for each variable should be a float")
		if len(self.constraints) != 1+self.n_vars+self.n_params:
			raise ValueError("Wrong number of constraints given. Required {} constraints for the variables but {} provided.".format(1+self.n_vars+self.n_params, len(self.constraints)))

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
		"""
		Calls to the NN. Inputs and outputs are tensorflow tensors.
		Input:
			X (None,1+n_vars+n_params)		Input values for the NN. Each row has the following layout: [t, X_0 (n_vars,), Omega (n_params,)]
		Output:
			out (None,n_vars)		Output of the network
		"""
		output = inputs
		for l in self._l_list:
			output = l(output)
		return output #(N,n_vars)

	def get_solution(self, inputs):
		"""
		Return the ansatz (NN + initial conditions) as:
			x(t;x_0,Omega) = x_0 + (1-exp(-t)) * NN(t,x_0,Omega)
		Everything is tensorflow.
		Input:
			X (None,1+n_vars+n_params)		Input values for the NN. Each row has the following layout: [t, X_0 (n_vars,), Omega (n_params,)]
		Output:
			X_out (None,n_vars)		Ansatz evaluated at X
		"""
		return inputs[:,1:self.n_vars+1] + tf.transpose(tf.math.multiply( tf.transpose(self.call(inputs)), 1-tf.math.exp(-1.*inputs[:,0])) ) #(N,n_vars)

	def __ok_inputs(self, inputs):
		if not isinstance(inputs, tf.Tensor):
			inputs = tf.convert_to_tensor(inputs, dtype=tf.float32) #(N,D)
			if inputs.ndim == 1:
				inputs = inputs[None,:]
		return inputs

	def ODE_derivative_np(self,t, x, Omega):
		"""
		Derivative of the ODE: F(t,x;Omega). Wrapper to ODE_derivative: it is suitable for scipy.integrate.
		Input:
			t (N,)					times to evaluate the derivative at
			x (N,n_var)				x values to evaluate the derivative at
			Omega (N, n_params)		Omega values to evaluate the derivative at
		Output:
			F (N, n_vars)		derivative of the ODE
		"""
		if x.ndim == 1:
			x = x[None,:]
			Omega = Omega[None,:]
			to_reshape = True
		res = self.ODE_derivative(tf.convert_to_tensor(t,dtype = tf.float32), tf.convert_to_tensor(x,dtype = tf.float32), tf.convert_to_tensor(Omega,dtype = tf.float32)).numpy()
		if to_reshape:
			return res[0,:]
		else:
			return res

	def ODE_solution(self,t, X_0, Omega):
		"""
		Numpy interface for the solution of the ODE with ML. Accepts a list of times, the initial conditions (n_vars,) and Omega (n_params,).
		Input:
			t (D,)				times to evaluate the solution at
			X_0 (n_vars,)		initial conditions
			Omega (n_params, )	parameters
		Output:
			X_t (D,n_vars)		ML solution to the ODE
		"""
		X_0 = np.array(X_0)
		Omega = np.array(Omega)
		assert X_0.shape == (self.n_vars,)
		assert Omega.shape == (self.n_params,)
		X = np.repeat([[*X_0, *Omega]],len(t), axis = 0) #(T,3)
		X = np.concatenate([np.array(t)[:, None],X], axis = 1) #(T,1+model.n_vars)
		X = self.__ok_inputs(X) #casting to tf
		res = self.get_solution(X) #(T,n_vars)
		return res.numpy()
	
	def loss(self, X):
		"""
		Loss function as a function of time, initial conditions and parameters. Input/outputs are tensorflow only.
		Input:
			X (None,1+n_vars+n_params)		values to test the model at. X[0,:] = [t, x_0 (n_vars,), Omega (n_params, ) ]
		Output:
			loss (None,)		values for the loss function
		"""
		Omega = X[:,self.n_vars+1:]
		with tf.GradientTape() as g:
			g.watch(X)
			out = self.get_solution(X)
		
		grad = g.batch_jacobian(out, X)[:,:,0] #d/dt #(N,3)
		F = self.ODE_derivative(X[:,0], out, Omega)
		if len(F.shape.as_list()) == 1:
			F = tf.expand_dims(F, 1) #adding a extra dimension for the tensor

			#loss can be multiplied by exp(-alpha*t) for "regularization"
		loss = tf.math.square(grad - F) #(N,n_vars)
		loss = tf.transpose(tf.math.multiply(tf.transpose(loss), tf.math.exp(-self.regularizer*X[:,0]))) #(N,n_vars)
		loss = tf.reduce_sum(loss, axis = 1) /X.shape[1] #(N,)
		return loss

	@tf.function #very useful for speed up
	def grad_update(self, X):
		"""
		Computes the gradients of the loss function w.r.t. NN's weights and performs the gradient update.
		Input should be tensorflow only.
		Input:
			X (None,1+n_vars+n_params)		values to test the model at. X[0,:] = [t, x_0 (n_vars,), Omega (n_params, ) ]
		Output:
			loss	scalar loss function averaged over a batch
		"""
		with tf.GradientTape() as g:
			g.watch(self.trainable_weights)
			loss = tf.reduce_sum(self.loss(X))/X.shape[0]

		gradients = g.gradient(loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

		return loss

	def get_random_X(self, N_batch, seed = None):
		"""
		Computes N_batch random values for the input X (1+n_vars+n_params) of the NN.
		Input:
			N_batch		number of batch to be extracted
			seed		random seed
		Output:
			X (N_batch,1+n_vars+n_params)		random values for X: X[0,:] = [t, x_0 (n_vars,), Omega (n_params, ) ]
		"""
		if isinstance(seed,int):
			tf.random.set_seed(seed)
		else:
			tf.random.set_seed(np.random.randint(0,1000000))

		random_Xs = []
		for _range in self.constraints:
			random_Xs.append(tf.random.uniform((N_batch,1), minval=_range[0], maxval=_range[1], dtype=tf.dtypes.float32))

		return tf.concat(random_Xs, axis = 1) #(N_batch, 7)
	
	def fit(self, N_epochs, N_batch = 20000,  learning_rate = 5e-4, save_output = True, save_step = 20000, print_step = 10, plot_function = None):
		"""
		Fit the model with gradient descent. It performs N_epochs updates of the gradients, each time using N_batch to compute the loss function. Occasionaly, it saves the model and computes the metric.
		It can save the ouput with a plot function. This can be any function with inputs an instance of the model and a path to save the output at.
		Input:
			N_epochs		number of training epochs
			N_batch			size of the batch to compute the loss funciton with
			learning_rate	learning rate for the training
			save_output		whether to occasionaly save the loss, the metric, the model and plotting the function
			save_step		step of training epochs at which to save the loss, the metric and to plot the results
			print_step		step of training epochs at which to print the loss and save the model
			plot_function	function for plotting. It must accept in input an instance of ML_ODE and a path to save the file at
		Output:
			history		history with the values of the loss function
		"""
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
					times = np.linspace(*self.constraints[0],100)
					for j in range(N_avg):
							#solving ODE for solution
						X_t = scipy.integrate.odeint(self.ODE_derivative_np, X[j,1:self.n_vars+1].numpy(), times, args = (X[j,self.n_vars+1:],), tfirst = True)

						X_t_NN = self.ODE_solution(times, X[j, 1:self.n_vars+1], X[j, self.n_vars+1:]) #(D,)
						metric += np.mean(np.square(X_t -X_t_NN))

					self.metric.append((self.epoch, metric/N_avg))
					print("\tMetric: {} {}".format(self.metric[-1][0],self.metric[-1][1]))

					self.save_weights("{}/{}/{}".format(self.name, str(self.epoch), self.name)) #saving to arxiv
					if plot_function is not None:
						plot_function(self, "{}/{}".format(self.name, str(self.epoch)))
						
					
		return self.history

	def load_everything(self, path):
		"""
		Loads model and tries to read metric and loss
		Input:
			path (str)		path to the model file
		"""
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
	"""
	Base mode for a function for plotting a comparison between the NN results and the actual solution.
	User can use this function to build function plot to input ML_ODE.fit()

	def plot(model, savefile):
		plot_solution(model, N_sol = 10, [0.], seed = 0, folder = savefile, show = False)

	Input:
		model		an instance of ML_ODE 
		N_sol		number of test solution to plot
		X_0			initial condition for all the test evaluations
		seed		random seed
		folder		folder at which the output is saved
		show		whether to show the saved plots
	"""
	X = model.get_random_X(N_sol, seed).numpy()
	X[:,1:1+model.n_vars] = X_0

	times = np.linspace(*model.constraints[0],200)
	X_t = np.zeros((N_sol, times.shape[0],model.n_vars))
	X_t_rec = np.zeros((N_sol, times.shape[0],model.n_vars))


	for i in range(N_sol):
		X_t_rec[i,:,:] = model.ODE_solution(times, X[i,1:1+model.n_vars], X[i,1+model.n_vars:]) #(N,2)
		X_t[i,:,:] = scipy.integrate.odeint(model.ODE_derivative_np, np.array(X[i,1:1+model.n_vars]), times, args = (np.array(X[i,1+model.n_vars:]),), tfirst = True)

	for var in range(model.n_vars):

		plt.figure()
		for i in range(N_sol):
			true, = plt.plot(times,X_t[i,:,var],  c = 'r')
			NN, = plt.plot(times,X_t_rec[i,:,var], c = 'b')
		plt.xlabel(r"$t$")
		plt.ylabel(r"$x_"+str(var)+"$")
		plt.legend([true, (true, NN)], ["True", "NN"])


		plt.savefig(folder+"/var{}.pdf".format(var), transparent =True)


	if show:
		plt.show()
	else:
		plt.close('all')

	return





