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
    v,_=momentum_to_velocity(p,m,c)
    p_minus = p + q * (E+vec_cross(v,B)) * dt / 2.0
    p_prime=p_minus +q*dt*E/2.0
    p_prime2 = vec_dot(p_prime, p_prime)

    gamma_prime = np.sqrt(1.0 + p_prime2 / (m**2 * c**2))

    tau = q * B * dt / 2.0
    tau2 = vec_dot(tau, tau)
    sigma=gamma_prime**2-tau2
    p_ast=vec_dot(p_prime,tau)/c
    p_ast2=vec_dot(p_ast,p_ast)
    gamma_plus=np.sqrt((sigma+np.sqrt(sigma**2+4*(tau2-p_ast2)))/2)
    t_vec=tau/gamma_plus
    t_vec2=vec_dot(t_vec,t_vec)
    s_factor = 1.0 / (1.0 + t_vec2)
    p_new=s_factor*(p_prime+vec_dot(p_prime,t_vec)*t_vec+vec_cross(p_prime,t_vec))
    return p_new


def momentum_to_velocity(p, m, c):
    p_sq = vec_dot(p, p)
    gamma = np.sqrt(1.0 + p_sq / (m**2 * c**2))
    v = p / (gamma * m)
    return v, gamma

# ---------- Bubble field (cleaned up) ----------
def bub(pos, t, vp):
    x, y, z = pos
    xi = z - vp * t

    Ex = E0 * kp *x / 4.0
    Ey = E0 * kp * y / 4.0
    Ez = E0 * kp * xi / 2.0

    # simple B consistent with E/c scaling (approx)
    Bx = E0*kp*y/(4*cst.c)
    By = -E0*kp*x /(4*cst.c)
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
vp=0.99*c
E0 = cst.electron_mass * wp * c /q 

kp=wp/c
w0 = 2.0 * np.sqrt(a0) / kp
# initial z position and time window (kept from your code)
xi_0 = -w0
gamma_p = 1.0 / np.sqrt(1 - vp**2 / c**2)

td = 2.0 * gamma_p**2 * w0 / c
tf = td / 2.0
ti = td / 1000.0
dt = td / 2000.0

# build time array robustly
T = np.linspace(ti, tf,num=int((tf - ti) / dt) + 1)
steps = len(T)

# initial momentum: choose initial gamma slightly >1 (avoid invalid sqrt)
initial_gamma = 100*gamma_p
print(gamma_p,initial_gamma)
pz0 = m*c*np.sqrt(initial_gamma**2 - 1)
print(pz0)
p = np.array([0, 0.0, pz0])   # initial momentum (px,py,pz)
x = np.array([-xi_0/4, 0.0, xi_0])  # position
# storage arrays
trajectory = np.zeros((steps, 3))
p_tot = np.zeros((steps, 3))
gamma_store = np.zeros(steps)
E_store = np.zeros((steps, 3))
gamma_z=np.zeros(steps)
for i, t in enumerate(T):
    trajectory[i] = x
    p_tot[i] = p
    v, gamma = momentum_to_velocity(p, m, c)
    _,gammaz=momentum_to_velocity(p[2],m,c)
    gamma_store[i] = gamma
    gamma_z[i]=gammaz
    # sample fields at particle position
    E, B = bub(x, t, vp)
    E_store[i] = E

    # push p, advance x
    p = RelativisticBorisPusher(p, E, B, dt, q, m, c)
    v, _ = momentum_to_velocity(p, m, c)
    x = x + v * dt

# relativistic kinetic energy
gamma_final = gamma_store
kinetic_energy = (gamma_final - 1.0) * m * c**2

# ---- Plots (same style as you intended) ----
fig, axs = plt.subplots(3, 2, figsize=(15, 10))
axs[0,0].plot(T*wp, kp*trajectory[:, 0])
axs[0,0].set_xlabel('t wp'); axs[0,0].set_ylabel('kp x'); axs[0,0].set_title('x vs time')
axs[0,1].plot(T*wp, kp*trajectory[:, 2])
axs[0,1].set_xlabel('t wp'); axs[0,1].set_ylabel('kp z'); axs[0,1].set_title('z vs time')
axs[1,0].plot(T*wp, p_tot[:, 0])
axs[1,0].set_xlabel('t (s)'); axs[1,0].set_ylabel('x (kg m/s)'); axs[1,0].set_title('px vs time')
axs[1,1].plot(T*wp, p_tot[:, 2])
axs[1,1].set_xlabel('t (s)'); axs[1,1].set_ylabel('pz (kg m/s)'); axs[1,1].set_title('pz vs time')
axs[2,0].plot(T*wp, gamma_store[:])
axs[2,0].set_xlabel('t (s)'); axs[2,0].set_ylabel('gamma'); axs[2,0].set_title('gamma_p vs time')
axs[2,1].plot(T*wp, gamma_z[:])
axs[2,1].set_xlabel('t (s)'); axs[2,1].set_ylabel('gamma_z'); axs[2,1].set_title('gamma_z vs time')
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
print(gamma_store[0],p_tot[0])
print("Final gamma:", gamma_store[-1])
print("Final momentum:", p_tot[-1])
print("Final position:", trajectory[-1])
