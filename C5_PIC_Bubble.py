import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst

# ---------- small helpers ----------
def vec_dot(A, B):
    # works for 1D arrays or shape (N,3) arrays with last axis being components
    return np.sum(A * B, axis=-1)

def vec_cross(A, B):
    return np.cross(A, B, axis=-1)

# ---------- Envelope (unchanged) ----------
def temp_env(prof: str, t, tau):
    if prof == "gauss":
        return np.exp(-(t / tau) ** 2)
    elif prof == "cst":
        return 1.0
    else:
        return 1.0

def spat_env(prof: str, x, L):
    if prof == "gauss":
        return np.exp(-(x / L) ** 2)
    elif prof == "cst":
        return 1.0
    else:
        return 1.0

# ---------- Relativistic Boris pusher (single particle, vector form) ----------
def RelativisticBorisPusher(p, E, B, dt, q, m, c):
    """
    p : (3,) momentum vector (SI)
    E, B : (3,) fields at particle position
    q, m, c : scalars
    returns new momentum p_new (3,)
    """
    # Half acceleration by E
    p_minus = p + (1 * E) * (dt / 2.0)

    # compute gamma from p_minus
    p_minus_sq = vec_dot(p_minus, p_minus)
    gamma_minus = np.sqrt(1.0 + p_minus_sq )

    # rotation coefficients
    t_vec = (1* B) * (dt / (2.0 * 1 * gamma_minus))
    t_sq = vec_dot(t_vec, t_vec)
    # p' = p_minus + p_minus x t
    p_prime = p_minus + vec_cross(p_minus, t_vec)
    # s factor
    s_factor = 2.0 / (1.0 + t_sq)
    # p_plus = p_minus + (p_prime x (s_factor * t_vec))
    p_plus = p_minus + vec_cross(p_prime, t_vec * s_factor)

    # final half-accel by E
    p_new = p_plus + (1 * E) * (dt / 2.0)
    return p_new

def momentum_to_velocity(p, m, c):
    p_sq = vec_dot(p, p)
    gamma = np.sqrt(1.0 + p_sq / (1**2 * 1**2))
    v = p / (gamma * 1)
    return v, gamma

# ---------- Bubble field (cleaned up) ----------
def bub(pos, t, vp):
    x, y, z = pos
    xi = z - vp * t

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(y/x)


    Ex = E0 * kp *x / 4.0
    Ey = E0 * kp * y / 4.0
    Ez = E0 * kp * xi / 2.0

    # simple B consistent with E/c scaling (approx)
    Bx = E0*kp*y/4
    By = -E0*kp*x /4
    Bz = 0.0

    E = np.array([Ex, Ey, Ez])
    B = np.array([Bx, By, Bz])
    return E, B

# ---------- parameters ----------
c = cst.c
q = -cst.e             # electron charge (negative)
m = cst.electron_mass
a0=5
A0=-m*c*a0/q
# n = 5e24
wp=np.sqrt(q**2*5e24/(m*cst.epsilon_0))
w=wp/a0
vp=0.99
kp=wp/c

a_0 = 5 
v_p = 0.99                                    # Phase velocity of plasma wave (99% speed of light)
beta_p = v_p                                 # Normalized phase velocity
gamma_p = 1/np.sqrt(1-beta_p**2)              # Lorentz factor of plasma wave                                
phi = np.pi/2 - 2*gamma_p                     # Phase parameter
phiprime = 0 
r_b = 2*np.sqrt(a_0)                                 # Bubble radius in meters (200 microns)
                                      # Normalized vector potential (laser strength)
ksi_i = -r_b                           # Initial position
gamma_d = (gamma_p**2)*(ksi_i**2)/2  
t_d = -2*gamma_p**2*ksi_i                   # Dephasing time scale
zeta_i = r_b/4


w0 = r_b
E0 = 1
tf = t_d
ti = t_d / 1000.0
dt = t_d / 2000.0
# initial z position and time window (kept from your code)
xi_0 = ksi_i


# build time array robustly
T = np.linspace(ti, tf,num=int((tf - ti) / dt) + 1)
steps = len(T)

# initial momentum: choose initial gamma slightly >1 (avoid invalid sqrt)
initial_gamma = 100*gamma_p
print(gamma_p,initial_gamma)
pz0 = initial_gamma
p = initial_gamma*np.array([0, 0.0, np.sqrt(1-1/(initial_gamma)**2)])   # initial momentum (px,py,pz)
x = np.array([r_b/4, 0.0, xi_0-vp*ti])  # position

# storage arrays
trajectory = np.zeros((steps, 3))
p_tot = np.zeros((steps, 3))
gamma_store = np.zeros(steps)
E_store = np.zeros((steps, 3))

for i, t in enumerate(T):
    trajectory[i] = x
    p_tot[i] = p
    v, gamma = momentum_to_velocity(p, 1, 1)
    gamma_store[i] = gamma

    # sample fields at particle position
    E, B = bub(x, t, vp)
    E_store[i] = E

    # push p, advance x
    p = RelativisticBorisPusher(p, E, B, dt, 1, 1, 1)
    v, _ = momentum_to_velocity(p, 1, 1)
    x = x + v * dt

# relativistic kinetic energy
gamma_final = gamma_store
kinetic_energy = (gamma_final - 1.0)

# ---- Plots (same style as you intended) ----
fig, axs = plt.subplots(3, 2, figsize=(15, 10))
axs[0,0].plot(T, trajectory[:, 0])
axs[0,0].set_xlabel('t wp'); axs[0,0].set_ylabel('kp x'); axs[0,0].set_title('x vs time')
axs[0,1].plot(T, trajectory[:, 2])
axs[0,1].set_xlabel('t wp'); axs[0,1].set_ylabel('kp z'); axs[0,1].set_title('z vs time')
axs[1,0].plot(T, p_tot[:, 0])
axs[1,0].set_xlabel('t (s)'); axs[1,0].set_ylabel('x (kg m/s)'); axs[1,0].set_title('px vs time')
axs[1,1].plot(T, p_tot[:, 2])
axs[1,1].set_xlabel('t (s)'); axs[1,1].set_ylabel('pz (kg m/s)'); axs[1,1].set_title('pz vs time')
axs[2,0].plot(T, gamma_store[:])
axs[2,0].set_xlabel('t (s)'); axs[2,0].set_ylabel('gamma'); axs[2,0].set_title('gamma_x vs time')
plt.tight_layout()
plt.show()

# 3D trajectory plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], lw=1)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)');
ax.set_title('Particle trajectory in bubble field')
plt.show()

# E-field at particle position
fig, ax = plt.subplots(figsize=(6,3))
ax.plot(T, E_store[:,0], label='Ex')
ax.plot(T, E_store[:,1], label='Ey')
ax.plot(T, E_store[:,2], label='Ez')
ax.set_xlabel('t (s)'); ax.set_ylabel('E (V/m)'); ax.set_title('E-field at particle position')
ax.legend()
plt.tight_layout()
plt.show()

print("Final gamma:", gamma_store[-1])
print("Final momentum:", p_tot[-1])
print("Final position:", trajectory[-1])
