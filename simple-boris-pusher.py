import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_e, epsilon_0
from numba import jit
from matplotlib.animation import FuncAnimation

#Coordinates
radius_est = 5e-3  # Estimated visible radius (meters)
x = np.linspace(0,2e-2,1000)
y = np.linspace(0,2e-2,1000)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

#Time parameters
T = np.linspace(0,1e-7,1000)  # Simulate up to 0.1 ns
dt = T[1] - T[0]

## Magnetic field only
B0 = 2e-3  # Magnetic field strength in T (smaller for larger radius)
@jit(nopython=True)
def B(x,y):
    return np.array([0, 0, B0])  # Uniform magnetic field in z-direction


#Initial conditions
r0 = np.array([radius_est, radius_est, 0.0])  # Initial position
v0 = np.array([1e6, 0.0, 0.0])  # Initial velocity (m/s)

#Iteration scheme
@jit(nopython=True)
def boris_push(r, v, q, m, dt):
    # Half acceleration due to electric field (E=0 here)
    v_minus = v

    # Rotation due to magnetic field
    t = (q * B(r[0], r[1]) / m) * (dt / 2)
    s = 2 * t / (1 + np.dot(t, t))
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)

    # Half acceleration due to electric field (E=0 here)
    v_new = v_plus

    # Update position
    r_new = r + v_new * dt
    print(r_new,v_new)
    return r_new, v_new


#Simulation
q = -e  # Charge of electron
m = m_e  # Mass of electron
r = r0
v = v0
positions = np.zeros((len(T), 3))
for i in range(len(T)):
    positions[i] = r
    r, v = boris_push(r, v, q, m, dt)

#Plotting

plt.figure(figsize=(8, 6))
plt.plot(positions[:, 0], positions[:, 1])
plt.xlim(0, 2e-2)
plt.ylim(0, 2e-2)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Trajectory of an Electron in a Uniform Magnetic Field')
plt.grid()
plt.show()

print(positions)

