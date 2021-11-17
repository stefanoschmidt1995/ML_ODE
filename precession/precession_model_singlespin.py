"""
This script uses the class ML_ODE to compute the time evolution of the Euler angles (in the single spin case).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'..')
from ML_ODE import *

#TODO: include in the loss function also the solution to the ODE... So the loss function is |f(t,y) - y'|^2 + |y-t_{true}|^2
#This should make things more accurate and easier to learn
#The inspiration is from: https://maziarraissi.github.io/research/1_physics_informed_neural_networks/

##Creating the basic class, which inherits from ML_ODE
class BBH_evolution_singlesping_model(ML_ODE_Basemodel):
	"Class for the BBH angle evolution - single spin"

	def __init__(self, name = "ML_ODE_model"):
		#we work in J0 frame
		super(BBH_evolution_singlesping_model, self).__init__(name)
		return

	def set_model_properties(self):
			#The independent variable is v (not the time)
			#we set M_tot = 1 M_sun everywhere

		self.regularizer = .4 #regularizer for the loss function (not compulsory, default 0)
		self.v_start = 0.25	#start at r == ??
		
		self.n_vars = 6	#Lhx, Lhy, Lhz, chi1x, chi1y, chi1z #number of variables in the problem
		self.n_params = 2 #(q, chi1) #number of parameters in the problem
		self.constraints = [(0.,.7), #constraints on v_shifted
			(-1.,1.), (-1.,1.), (-1.,1.), (-1.,1.), (-1.,1.), (-1.,1.), 
			(1.,3.), (0.,.8)]

	def build_NN(self):
		#self._l_list.append(tf.keras.layers.Dense(128*8, activation=tf.nn.tanh) )
		self._l_list.append(tf.keras.layers.Dense(128*2, activation=tf.nn.tanh) )
		self._l_list.append(tf.keras.layers.Dense(128, activation=tf.nn.tanh) )
		self._l_list.append(tf.keras.layers.Dense(self.n_vars, activation=tf.keras.activations.linear))
	
	def get_random_X(self, N_batch, seed = None):
		"Enforce the initial condition by drawing a random input for the network"
		#TODO: enforce the J0 frame here!! Solution will be smoother

		if isinstance(seed,int):
			tf.random.set_seed(seed)
		else:
			tf.random.set_seed(np.random.randint(0,1000000))

		params_list = [] #(q, chi1)
		for _range in self.constraints[1+self.n_vars:]:
			params_list.append(tf.random.uniform((N_batch,), minval=_range[0], maxval=_range[1], dtype=tf.dtypes.float32)) #(N_batch,)
		
			#extracting spin angles
			##########
			#setting chi1
		theta1 = tf.random.uniform((N_batch,), minval= 0, maxval= np.pi, dtype=tf.dtypes.float32)
		phi1 = tf.random.uniform((N_batch,), minval= 0, maxval= 2*np.pi, dtype=tf.dtypes.float32)

		chi1x = tf.math.sin(theta1)*tf.math.cos(phi1)
		chi1y = tf.math.sin(theta1)*tf.math.sin(phi1)
		chi1z = tf.math.cos(theta1)

			#computing L, s.t. the system is in J_0 frame
		M =1.
		r = float(M /self.v_start**2)
		q = params_list[0]
		L= ((q/(1.+q)**2)*(r*M**3)**.5) #(N,)
		S = tf.stack([chi1x, chi1y, chi1z], axis = -1)*params_list[1][:,None]*tf.square(q*M/(1+q))[:,None] #(N,3)
		S_norm = tf.norm(S,axis = 1) #(N,)
		cos_phi_JS = S[:,2]/S_norm #(N,)
			
			# L = J - S -> L^2 = J^2+S^2-2|J||S|cos(phi_JS)
		J_norm = S_norm * cos_phi_JS + tf.sqrt(S_norm**2*(cos_phi_JS**2-1)+L**2) #(N,)
		J = tf.einsum('i,j->ij', J_norm, tf.constant([0.,0.,1.], dtype=tf.dtypes.float32))
		L_hat = J - S #actually L
			#print("norm L_hat", L,  tf.norm(L_hat,axis = 1) ) #The two norms agree!!
			
		L_hat = tf.transpose(tf.transpose(L_hat)/L)
			
		random_Xs = [tf.random.uniform((N_batch,), minval= self.constraints[0][0], maxval= self.constraints[0][1], dtype=tf.dtypes.float32), #v_shifted extraction
			L_hat[:,0], L_hat[:,1], L_hat[:,2]] #setting L in J0 frame
		
		random_Xs.extend([chi1x, chi1y, chi1z])
		random_Xs.extend(params_list)
		
		to_return = tf.stack(random_Xs, axis = -1)
		
		return to_return
			
	
	def ODE_derivative(self, v_shifted, allvars, params): 
		"""
		Derivative of the model
		"""
		#see: https://github.com/dgerosa/precession/blob/1e121ae382f0af478aca602c04c694eeaeb189c9/precession/precession.py#L3489
		#FIXME: check that the integration is OK! It doesn't seem so
			#v_start -> initial v of the model
			#Here we are forced to set v = 0 at initial time => traslation in v
			#v = (M/r)**0.5 -> r is the BBH distance

		v = v_shifted+self.v_start

			#Setting parameters
		q = 1./params[:,0] #q is less than 1 in this shitty convention

		M = 1.
		eta = q/(1+q)**2
		m1 = M/(1+q)
		m2 = q*M/(1+q)
		
			# Read variables in
		Lhx=allvars[:,0]
		Lhy=allvars[:,1]
		Lhz=allvars[:,2]
		S1hx=allvars[:,3]
		S1hy=allvars[:,4]
		S1hz=allvars[:,5]
		chi1 = params[:,1] #tf.sqrt(S1hx**2+ S1hy**2 + S1hz**2)

			#FIXME: you make the assumption that the magnitude of the spin remains constant during the evolution. Can you do better?
		S1 = chi1*m1**2

			#normalizing angular momentum and spins (might be useful)
			#TODO: understand whether you should normalize here or not...
		if True:
			L = tf.sqrt(Lhx**2+ Lhy**2 + Lhz**2)
			Lhx = Lhx/L
			Lhy = Lhy/L
			Lhz = Lhz/L

				#FIXME: check on 1e-10
			norm1 = tf.sqrt(S1hx**2+ S1hy**2 + S1hz**2) + 1e-10
			S1hx = S1hx /norm1
			S1hy = S1hy /norm1
			S1hz = S1hz /norm1

		# Useful variables
		ct1=(Lhx*S1hx+Lhy*S1hy+Lhz*S1hz)
		ct2= 0.
		ct12= 0.

		# Spin precession for S1
		Omega1x= eta*v**5*(2.+3.*q/2.)*Lhx/M  \
            + v**6*(-3.*q*S1*ct1*Lhx)/(2.*M**3)
		Omega1y= eta*v**5*(2.+3.*q/2.)*Lhy/M  \
            + v**6*(-3.*q*S1*ct1*Lhy)/(2.*M**3)
		Omega1z= eta*v**5*(2.+3.*q/2.)*Lhz/M  \
            + v**6*(-3.*q*S1*ct1*Lhz)/(2.*M**3)

		dS1hxdt= Omega1y*S1hz - Omega1z*S1hy
		dS1hydt= Omega1z*S1hx - Omega1x*S1hz
		dS1hzdt= Omega1x*S1hy - Omega1y*S1hx

		# Conservation of angular momentum
		dLhxdt= -1.*v*(S1*dS1hxdt)/(eta*M**2)
		dLhydt= -1.*v*(S1*dS1hydt)/(eta*M**2)
		dLhzdt= -1.*v*(S1*dS1hzdt)/(eta*M**2)

		# Radiation reaction
		quadrupole_formula=False
		dvdt= (32.*eta*v**9/(5.*M))
		
		dtdv=1./dvdt
		dLhxdv=dLhxdt*dtdv
		dLhydv=dLhydt*dtdv
		dLhzdv=dLhzdt*dtdv
		dS1hxdv=dS1hxdt*dtdv
		dS1hydv=dS1hydt*dtdv
		dS1hzdv=dS1hzdt*dtdv
		
		to_return = tf.stack([dLhxdv, dLhydv, dLhzdv, dS1hxdv, dS1hydv, dS1hzdv], axis =1)

		return to_return



def plot(model, savefile, show = True):
	X_0 = model.get_random_X(1, seed = 0).numpy()[1:1+model.n_vars]
	plot_solution(model, 3, X_0, seed = 0, folder = savefile, show = show)

################################
#######start of the juicy part
################################

if __name__ == '__main__':
	#Building and fitting the model
	what_to_do = "fit"

	tf.debugging.experimental.enable_dump_debug_info(
    "./tfdbg2_logdir",
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)

	model_name = "p_model_singlespin"
	model = BBH_evolution_singlesping_model(model_name)
	model_file = "{}/{}".format(model_name, model_name) #file which holds the model

	print('BBH_evolution_singlesping_model')
	print(model.summary())

		#deciding what to do
	if what_to_do == 'load':
		model.load_everything(model_file)
	elif what_to_do == 'fitload' or what_to_do == 'fit':
		if what_to_do == 'fitload':
			model.load_everything(model_file)
		model.fit(100000, N_batch = 10000,  learning_rate = 1e-4, save_output = True, plot_function = plot, save_step = 10000, print_step = 20)
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










