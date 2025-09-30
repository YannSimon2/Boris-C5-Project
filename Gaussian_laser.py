import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------- Math helpers ----------
def dot(V,B):
    return np.sum(V*B, axis=1)

def cross(A,B):
    res = np.array([
        A[:,1]*B[:,2] - A[:,2]*B[:,1],
        A[:,2]*B[:,0] - A[:,0]*B[:,2],
        A[:,0]*B[:,1] - A[:,1]*B[:,0]
    ])
    return res.T

# ---------- Enveloppe ----------
def temp_env(prof:str,t,tau):
    if prof=="gauss":
        return np.exp(-(t/tau)**2)
    elif prof=="cst":
        return 1.0


# ---------- Relativistic Boris ----------
def RelativisticBorisPusher(p, E, B, dt, q, m, c=1.0):
    p = np.atleast_2d(p)
    E = np.atleast_2d(E)
    B = np.atleast_2d(B)

    N = p.shape[0]
    q = np.full(N, q) if np.ndim(q)==0 else np.asarray(q)
    m = np.full(N, m) if np.ndim(m)==0 else np.asarray(m)

    qm = (q/m)[:,None]
    p_minus = p + qm * E * (dt/2.0)

    p_mc = p_minus / (m[:,None] * c)
    gamma_minus = np.sqrt(1.0 + dot(p_mc,p_mc)[:,None])

    t = (q[:,None] * B) * (dt / (2.0 * m[:,None] * gamma_minus))
    t2 = dot(t,t)[:,None]
    s = 2.0 * t / (1.0 + t2)

    p_prime = p_minus + cross(p_minus, t)
    p_plus  = p_minus + cross(p_prime, s)

    p_new = p_plus + qm * E * (dt/2.0)
    return p_new

def momentum_to_velocity(p, m, c=1.0):
    p = np.atleast_2d(p)
    N = p.shape[0]
    m = np.full(N, m) if np.ndim(m)==0 else np.asarray(m)
    p_mc = p / (m[:,None] * c)
    gamma = np.sqrt(1.0 + dot(p_mc,p_mc)[:,None])
    v = p / (gamma * m[:,None])
    return v

# ---------- Gaussian Laser Pulse ----------
def gaussian_laser_field(x, t, E0, w0, tau,p,pr,k, omega, c,profil:str):
    """
    Linearly polarized Gaussian laser along x
    propagating in +z direction.
    - w0  : waist (transverse Gaussian)
    - tau : pulse duration (temporal Gaussian)
    - p: polarisation (0=x, 1=y, 2=z)
    - pr: propagation (0=x, 1=y, 2=z)
    """
    if pr==0:
        z = x[:,0]
        r2 = x[:,2]**2 + x[:,1]**2
    elif pr==1:
        z = x[:,1]
        r2 = x[:,2]**2 + x[:,0]**2
    elif pr==2:
        z = x[:,2]
        r2 = x[:,0]**2 + x[:,1]**2
    env_t = temp_env(prof=profil,t=t,tau=tau)
    env_r = np.exp(- r2 / (w0**2))
    phase = k*z - omega*t
    Ep = E0 * env_t * env_r * np.cos(phase)
    # Magnetic field from plane wave (Ey=0, B_y ~ Ex/c)
    Bp = Ep / c
    if p==0:
        E = np.stack([Ep, np.zeros_like(Ep), np.zeros_like(Ep)], axis=1)
        B = np.stack([np.zeros_like(Bp), Bp, np.zeros_like(Bp)], axis=1)
    elif p==1:
        E = np.stack([np.zeros_like(Ep),Ep, np.zeros_like(Ep)], axis=1)
        B = np.stack([np.zeros_like(Bp), np.zeros_like(Bp),Bp], axis=1)
    elif p==2:
        E = np.stack([np.zeros_like(Ep), np.zeros_like(Ep),Ep], axis=1)
        B = np.stack([Bp,np.zeros_like(Bp), np.zeros_like(Bp)], axis=1)  
    return E, B

# ---------- Simulation Parameters ----------
c   = 1.0
q   = -1.0      # electron charge sign
m   = 1.0
dt  = 0.05
steps =10000
w=1
k=1/w
lamb=2*np.pi*k
I0=10            #10^18
A0=0.85*np.sqrt(I0)
propag=2
w0=5
polar=1
# initial momentum and position
p = np.array([[0.0, 0.0, 0.0]])
x = np.array([[0.0, 0.0, 0.0]])   # start before the pulse

trajectory = np.zeros((steps+1, 3))
trajectory[0] = x[0]

# ---------- Time loop ----------
for i in range(steps):
    t = i*dt
    E, B = gaussian_laser_field(x, t,E0=A0,w0=5.0,tau=20.0,p=polar,pr=propag,k=k,omega=w,c=c,profil='gauss')
    p = RelativisticBorisPusher(p, E, B, dt, q, m, c=c)
    v = momentum_to_velocity(p, m, c=c)
    x = x + v * dt
    trajectory[i+1] = x[0]

# ---------- Plot the 3D trajectory ----------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], lw=1.0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Relativistic Electron in a Gaussian Laser Pulse')
plt.show() 

## ajouter champ B exterieur, multiple particule (zone avec densité etc) et profils lasers

# Bubble regime

