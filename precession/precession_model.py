"""
This script uses the class ML_ODE to compute the time evolution of the Euler angles.
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
class BBH_evolution_model(ML_ODE_Basemodel):
	"Class for the cosmological model."

	def __init__(self, name = "ML_ODE_model", zero_s2= False, J0_frame = False):
		self.zero_s2 = zero_s2
		self.J0_frame = J0_frame
		super(BBH_evolution_model, self).__init__(name)
		return

	def set_model_properties(self):
			#The independent variable is v (not the time)
			#we set M_tot = 1 M_sun everywhere

		self.regularizer = .4 #regularizer for the loss function (not compulsory, default 0)
		self.v_start = 0.25	#start at r == ??
		
		if not self.zero_s2:
			self.n_vars = 9	#Lhx, Lhy, Lhz, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z #number of variables in the problem
			self.n_params = 3 #(q, chi1, chi2) #number of parameters in the problem
			self.constraints = [(0.,.7), #constraints on v_shifted
				(.0, .0), (.0, .0), (1.,1.), (-0.57,0.57), (-0.57,0.57), (-0.57,0.57), (-0.57,0.57), (-0.57,0.57), (-0.57,0.57),
				(1.,3.), (0.,1.), (0.,1.)]
		
		if self.zero_s2:
			self.n_vars = 6	#Lhx, Lhy, Lhz, chi1x, chi1y, chi1z #number of variables in the problem
			self.n_params = 2 #(q, chi1, chi2) #number of parameters in the problem
			self.constraints = [(0.,.7), #constraints on v_shifted
				(.0, .0), (.0, .0), (1.,1.), (-0.57,0.57), (-0.57,0.57), (-0.57,0.57), 
				(1.,3.), (0.,.8)]

	def build_NN(self):
		self._l_list.append(tf.keras.layers.Dense(128*8, activation=tf.nn.tanh) )
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

		params_list = [] #(q,chi1, chi2)/(q, chi1)
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

			#setting chi2 (only if not zero_s2)
		if not self.zero_s2:
			theta2 = tf.random.uniform((N_batch,), minval= 0, maxval= np.pi, dtype=tf.dtypes.float32)
			phi2 = tf.random.uniform((N_batch,), minval= 0, maxval= 2*np.pi, dtype=tf.dtypes.float32)
			chi2x = tf.math.sin(theta2)*tf.math.cos(phi2)*int(not self.zero_s2)
			chi2y = tf.math.sin(theta2)*tf.math.sin(phi2)*int(not self.zero_s2)
			chi2z = tf.math.cos(theta2)*int(not self.zero_s2)
		
		
			#computing L, s.t. the system is in J_0 frame
		if self.J0_frame:
			M =1.
			r = float(M /self.v_start**2)
			q = params_list[0]
			L= ((q/(1.+q)**2)*(r*M**3)**.5) #(N,)
			S = tf.stack([chi1x, chi1y, chi1z], axis = -1)*params_list[1][:,None]*tf.square(q*M/(1+q))[:,None] #(N,3)
			if not self.zero_s2:
				S = S + tf.stack([chi2x, chi2y, chi2z], axis = -1)*params_list[2][:,None]*tf.square(M/(1+q))[:,None] #(N,3)
			S_norm = tf.norm(S,axis = 1) #(N,)
			cos_phi_JS = S[:,2]/S_norm #(N,)
			
			# L = J - S -> L^2 = J^2+S^2-2|J||S|cos(phi_JS)
			#FIXME: check this!!
			J_norm = S_norm * cos_phi_JS + tf.sqrt(S_norm**2*(cos_phi_JS**2-1)+L**2) #(N,)
			J = tf.einsum('i,j->ij', J_norm, tf.constant([0.,0.,1.], dtype=tf.dtypes.float32))
			L_hat = J - S
			#print("norm L_hat", L,  tf.norm(L_hat,axis = 1) ) #The two norms agree!!
			
			L_hat = tf.transpose(tf.transpose(L_hat)/L)
			
			random_Xs = [tf.random.uniform((N_batch,), minval= self.constraints[0][0], maxval= self.constraints[0][1], dtype=tf.dtypes.float32), #v_shifted extraction
				L_hat[:,0], L_hat[:,1], L_hat[:,2]] #setting L in J0 frame
		else:
			random_Xs = [tf.random.uniform((N_batch,), minval= self.constraints[0][0], maxval= self.constraints[0][1], dtype=tf.dtypes.float32), #v_shifted extraction
				tf.zeros((N_batch,)), tf.zeros((N_batch,)), tf.ones((N_batch,))] #setting the L_0 = (0,0,1) frame
		
		if self.zero_s2:
			random_Xs.extend([chi1x, chi1y, chi1z])
		else:
			random_Xs.extend([chi1x, chi1y, chi1z, chi2x, chi2y, chi2z])
		random_Xs.extend(params_list)
		to_return = tf.stack(random_Xs, axis = -1)
		#print(to_return[0,:], to_return.shape)
		
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
		if not self.zero_s2:
			chi2 = params[:,2]#tf.sqrt(S2hx**2+ S2hy**2 + S2hz**2)		
			S2hx=allvars[:,6]
			S2hy=allvars[:,7]
			S2hz=allvars[:,8]
		else:
			chi2 = tf.zeros(params[:,1].shape) 
			S2hx=tf.zeros(allvars[:,5].shape)
			S2hy=tf.zeros(allvars[:,5].shape)
			S2hz=tf.zeros(allvars[:,5].shape)

			#FIXME: you make the assumption that the magnitude of the spin remains constant during the evolution. Can you do better?
		S1 = chi1*m1**2
		S2 = chi2*m2**2

			#normalizing angular momentum and spins (might be useful)
			#TODO: understand whether you should normalize here or not...
		if False:
			L = tf.sqrt(Lhx**2+ Lhy**2 + Lhz**2)
			Lhx = Lhx/L
			Lhy = Lhy/L
			Lhz = Lhz/L

				#FIXME: check on 1e-10
			norm1 = tf.sqrt(S1hx**2+ S1hy**2 + S1hz**2) + 1e-10
			norm2 = tf.sqrt(S2hx**2+ S2hy**2 + S2hz**2) +1e-10
			S1hx = S1hx /norm1
			S1hy = S1hy /norm1
			S1hz = S1hz /norm1
			S2hx = S2hx /norm2
			S2hy = S2hy /norm2
			S2hz = S2hz /norm2

		# Useful variables
		ct1=(Lhx*S1hx+Lhy*S1hy+Lhz*S1hz)
		ct2=(Lhx*S2hx+Lhy*S2hy+Lhz*S2hz)
		ct12=(S1hx*S2hx+S1hy*S2hy+S1hz*S2hz)

		# Spin precession for S1
		Omega1x= eta*v**5*(2.+3.*q/2.)*Lhx/M  \
            + v**6*(S2*S2hx-3.*S2*ct2*Lhx-3.*q*S1*ct1*Lhx)/(2.*M**3)
		Omega1y= eta*v**5*(2.+3.*q/2.)*Lhy/M  \
            + v**6*(S2*S2hy-3.*S2*ct2*Lhy-3.*q*S1*ct1*Lhy)/(2.*M**3)
		Omega1z= eta*v**5*(2.+3.*q/2.)*Lhz/M  \
            + v**6*(S2*S2hz-3.*S2*ct2*Lhz-3.*q*S1*ct1*Lhz)/(2.*M**3)

		dS1hxdt= Omega1y*S1hz - Omega1z*S1hy
		dS1hydt= Omega1z*S1hx - Omega1x*S1hz
		dS1hzdt= Omega1x*S1hy - Omega1y*S1hx

		# Spin precession for S2
		Omega2x= eta*v**5*(2.+3./(2.*q))*Lhx/M  \
            + v**6*(S1*S1hx-3.*S1*ct1*Lhx-3.*S2*ct2*Lhx/q)/(2.*M**3)
		Omega2y= eta*v**5*(2.+3./(2.*q))*Lhy/M  \
            + v**6*(S1*S1hy-3.*S1*ct1*Lhy-3.*S2*ct2*Lhy/q)/(2.*M**3)
		Omega2z= eta*v**5*(2.+3./(2.*q))*Lhz/M  \
            + v**6*(S1*S1hz-3.*S1*ct1*Lhz-3.*S2*ct2*Lhz/q)/(2.*M**3)

		dS2hxdt= Omega2y*S2hz - Omega2z*S2hy
		dS2hydt= Omega2z*S2hx - Omega2x*S2hz
		dS2hzdt= Omega2x*S2hy - Omega2y*S2hx

		# Conservation of angular momentum
		dLhxdt= -1.*v*(S1*dS1hxdt+S2*dS2hxdt)/(eta*M**2)
		dLhydt= -1.*v*(S1*dS1hydt+S2*dS2hydt)/(eta*M**2)
		dLhzdt= -1.*v*(S1*dS1hzdt+S2*dS2hzdt)/(eta*M**2)

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
		dS2hxdv=dS2hxdt*dtdv
		dS2hydv=dS2hydt*dtdv
		dS2hzdv=dS2hzdt*dtdv
		
		
		if self.zero_s2:
			to_return = tf.stack([dLhxdv, dLhydv, dLhzdv, dS1hxdv, dS1hydv, dS1hzdv], axis =1)
		else:
			to_return = tf.stack([dLhxdv, dLhydv, dLhzdv, dS1hxdv, dS1hydv, dS1hzdv, dS2hxdv, dS2hydv, dS2hzdv], axis =1)
		#to_return = tf.stack([dLhxdv, dLhydv, dLhzdv, dS1hxdv*chi1, dS1hydv*chi1, dS1hzdv*chi1, dS2hxdv*chi2, dS2hydv*chi2, dS2hzdv*chi2], axis =1) #this is for the case you include the spin

		print(allvars[:2,:], tf.linalg.norm(allvars[:2,1:4],axis =1), params[:2,:])
		#print(to_return[:2,:].numpy())
		
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

	model_name = "p_model_Jframe_s20"
	zero_s2 = True
	J0_frame = True
	print('zero_s2: {}\nJ0_frame: {}'.format(zero_s2, J0_frame))
	model = BBH_evolution_model(model_name, zero_s2, J0_frame)
	model_file = "{}/{}".format(model_name, model_name) #file which holds the model

	print(model.summary())

		#deciding what to do
	if what_to_do == 'load':
		model.load_everything(model_file)
	elif what_to_do == 'fitload' or what_to_do == 'fit':
		if what_to_do == 'fitload':
			model.load_everything(model_file)
		model.fit(100000, N_batch = 10000,  learning_rate = 1e-4, save_output = True, plot_function = plot, save_step = 2000, print_step = 5)
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










