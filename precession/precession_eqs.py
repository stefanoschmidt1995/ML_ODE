from precession import *
import os
import numpy as np
import matplotlib.pyplot as plt
#from precession_model import *
from precession_model_singlespin import *

zero_s2 = True
J0_frame = True
#model = BBH_evolution_model('prec_model', zero_s2, J0_frame)
model = BBH_evolution_singlesping_model('prec_model_singlespin')
X_0 = model.get_random_X(1, None).numpy()[0,1:] #extracting at random (and removing time)

print("Random initial conditions: ", X_0)

if zero_s2:
	print(X_0.shape, X_0[model.n_vars:].shape)
	q, chi1 = X_0[model.n_vars:]
	chi2 = 1e-10
else:
	q, chi1, chi2 = X_0[model.n_vars:]
M = 1.
m1 = q*M/(1+q)
m2 = M/(1+q)

v_vals = np.linspace(model.v_start, 1., 10000) #v = (M/r)**0.5
r_vals = [M/v**2 for v in v_vals]
print("Initial frequency: ", rtof(r_vals[0], 1))

L_0 = (q/(1.+q)**2)*(r_vals[0]*M**3)**.5

Lvec = X_0[:3] * L_0
S1vec = X_0[3:6]*chi1*m1**2
if zero_s2:
	S2vec = np.array([1e-10,1e-10,1e-10])
else:	
	S2vec = X_0[6:9]*chi2*m2**2 + 1e-10

print("L0", L_0)

Lx_fvals, Ly_fvals, Lz_fvals, S1x_fvals, S1y_fvals, S1z_fvals, S2x_fvals, S2y_fvals, S2z_fvals = precession.orbit_vectors(*Lvec, *S1vec, *S2vec, r_vals, 1./q, time = False)
os.system('rm -r precession_checkpoints/')

X_out = np.stack([Lx_fvals, Ly_fvals, Lz_fvals, S1x_fvals, S1y_fvals, S1z_fvals, S2x_fvals, S2y_fvals, S2z_fvals], axis = 1)
X_out[:,:3] = (X_out[:,:3].T/np.linalg.norm(X_out[:,:3], axis =1)).T

v_tf, X_out_tf = solve_ode_numerically(model, X_0[:model.n_vars], X_0[model.n_vars:], N_points = len(v_vals), show = False)
X_out_tf[:,3:6] = X_out_tf[:,3:6]*chi1*m1**2
if not zero_s2: X_out_tf[:,6:] = X_out_tf[:,6:]*chi2*m2**2

for var in range(X_out_tf.shape[1]):
	plt.figure()
	plt.title("ODE solution\nInitial condition: {}\nParams: {}".format(X_0, q))
	true, = plt.plot(v_vals, X_out[:,var],  c = 'r')
	tf, = plt.plot(v_tf+model.v_start, X_out_tf[:,var], c = 'b')
	plt.xlabel(r"$t$")
	plt.ylabel(r"$x_"+str(var)+"$")

#plt.figure()
#plt.title("S1")
#plt.plot(v_vals, np.linalg.norm(X_out[:,3:6],axis =1))

plt.show()

