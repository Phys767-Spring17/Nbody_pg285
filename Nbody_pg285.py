import numpy as np
import time
import matplotlib.pyplot as plt

'''
N = number of bodies
D = dimension
x = position; x_i = position in x, x_j = position in y
v = velocity
G = gravity
m = mass
'''

#Generates initial conditions for N unity masses at rest starting at random positions in D-dimensional space
#	returns a tuple composed of arrays, where the x and v are NxD arrays and m is 1D array.
#x0 generates one array with N number of arrays that each have D number of random floats less than 1
#v0 generates one array with N number of arrays that each have D number of zero floats
#m generates one array with N number of 1 floats
def initial_cond(N, D):	
	x0 = np.random.rand(N, D)
	v0 = np.zeros((N, D), dtype=float)
	m = np.ones(N, dtype=float)
	return x0, v0, m

#Function drops the ith element of an array.
def remove_ith(x, i): 
	shape = (x.shape[0]-1,) + x.shape[1:]
	y = np.empty(shape, dtype=float)
	y[:i] = x[:i]
	y[i:] = x[i+1:]
	return y

def a(i, x, G, m):
	"""The acceleration of the ith mass."""
	x_i = x[i]
	x_j = remove_i(x, i)
	m_j = remove_i(m, i)
	diff = x_j - x_i
	mag3 = np.sum(diff**2, axis=1)**1.5
	result = G * np.sum(diff * (m_j / mag3)[:,np.newaxis], axis=0)#np.newaxis increases the array by 1 dimension. 1D array with 3 items becomes a 2D array with 3 rows of 1 item
	return result
	
def timestep(x0, v0, G, m, dt):	
	"""Computes the next position and velocity for all masses given
	initial conditions and a time step size.
	"""
	N = len(x0)
	x1 = np.empty(x0.shape, dtype=float) #same shape of entries as x0, all empty except last entry
	v1 = np.empty(v0.shape, dtype=float) #same shape of entries as v0, all empty except last entry
	for i in range(N):
		a_i0 = a(i, x0, G, m) 
		v1[i] = a_i0 * dt + v0[i]
		x1[i] = a_i0 * dt**2 + v0[i] * dt + x0[i]
	return x1, v1

#Generates initial conditions for N unity masses at rest starting at random positions in D-dimensional space.
def initial_cond(N, D):		#first function used
	x0 = np.random.rand(N, D) #generates one array with N number of arrays that each have D number of random floats less than 1
	v0 = np.zeros((N, D), dtype=float) #generates one array with N number of arrays that each have D number of zero floats
	m = np.ones(N, dtype=float) #generates one array with N number of 1 floats
	return x0, v0, m


#x0, v0, m = initial_cond(10, 2)
#x1, v1 = timestep(x0, v0, 1.0, m, 1.0e-3)

'''def simulate(N, D, S, G, dt):
	x0, v0, m = initial_cond(N, D)
	x=[];y=[]
	for s in range(S):
		x1, v1 = timestep(x0, v0, G, m, dt)
		x0, v0 = x1, v1'''


def simulate(N, D, S, G, dt):
	x0, v0, m = initial_cond(N, D)
	x=[];v=[]
	for s in range(S):
		x1, v1 = timestep(x0, v0, G, m, dt)
		x.append(x1),v.append(v1)
		x0, v0 = x1, v1
	return x,v

'''
x,v=simulate(10,2,5,1.0,1e-2)

color=['b','m','r','k','g']
for i in range(len(x)):
	for j in range(len(x[i])):
		plt.scatter(x[i][j][0],x[i][j][1], color=color[i], marker='o',s=50)
		plt.title('Time '+str(i), y=1.05, fontsize=17, fontweight='bold')
		plt.ylabel('y', fontsize=20, fontweight='bold')
		plt.xlabel('x', fontsize=20, fontweight='bold')
		plt.axis([0,1.5,0,1.5],fontsize=30, fontweight='bold')
	plt.show()


print x1
for i in range(len(x1)):
	plt.scatter(x1[i][0],x1[i][1])
plt.show()

Ns = [2, 4, 8, 16, 32, 64, 128]#, 256, 512, 1024, 2048, 4096, 8192]
runtimes = []
for N in Ns:
	start = time.time()
	simulate(N, 3, 300, 1.0, 1e-3)
	stop = time.time()
	runtimes.append(stop - start)

#print runtimes'''

#___________________________________________________________________________________________
#np.newaxis
'''x1 = np.array([5, 6, 7, 8, 9])
x2 = np.array([5, 4, 3])
x1_new = x1[:, np.newaxis]
print x1_new
print ''
print x1_new + x2'''
