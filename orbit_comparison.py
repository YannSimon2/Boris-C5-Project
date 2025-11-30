import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.constants as const

#------------------------------------ Plotting functions --------------------------------#

def comparaison(L_t, L_r, L_analytic, relativistic = False, save=False):
    """
    Plots the numerical and analytical trajectories and their relative errors for each axis.
    Args:
        L_t: Array of time steps
        L_r: Numerical positions
        L_analytic: Analytical positions
        save: If True, saves the plot as a PNG file
    """
    if relativistic:
        fig = plt.figure(figsize=(12, 7))
        gs = gridspec.GridSpec(2, 3, fig, height_ratios=[2, 1], hspace=0)
        fig.suptitle("Trajectory and error on each axis for a relativistic particle")

        ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
        ax3, ax4 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
        ax5, ax6 = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
    else:
        fig = plt.figure(figsize=(12, 7))
        gs = gridspec.GridSpec(2, 3, fig, height_ratios=[2, 1], hspace=0)
        fig.suptitle("Trajectory and error on each axis for a non-relativistic particle")

        ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
        ax3, ax4 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
        ax5, ax6 = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])

    # X component
    ax1.set_title("X component")
    ax1.plot(L_t, L_r[:, 0], label="Numerical", color="blue")
    ax1.plot(L_t, L_analytic[:, 0], "--", label="Analytic", color="red")
    ax1.set_xticks([])
    ax1.set_ylabel("Displacement")
    ax1.legend()
    ax2.plot(L_t, np.abs((L_r[:, 0] - L_analytic[:, 0]) / L_analytic[:, 0]) * 100, color="green")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Relative error [%]")

    # Y component
    ax3.set_title("Y component")
    ax3.plot(L_t, L_r[:, 1], label="Numerical", color="blue")
    ax3.plot(L_t, L_analytic[:, 1], "--", label="Analytic", color="red")
    ax3.set_xticks([])
    ax3.legend()
    ax4.plot(L_t, np.abs((L_r[:, 1] - L_analytic[:, 1]) / L_analytic[:, 1]) * 100, color="green")
    ax4.set_xlabel("Time")

    # Z component
    ax5.set_title("Z component")
    ax5.plot(L_t, L_r[:, 2], label="Numerical", color="blue")
    ax5.plot(L_t, L_analytic[:, 2], "--", label="Analytic", color="red")
    ax5.set_xticks([])
    ax5.legend()
    ax6.plot(L_t, np.abs((L_r[:, 2] - L_analytic[:, 2]) / L_analytic[:, 2]) * 100, color="green")
    ax6.set_xlabel("Time")

    plt.tight_layout()
    if relativistic:
        if save:
            plt.savefig("image_comparaison/comparaison_relativistic.png", dpi=300)
    else:
        if save:
            plt.savefig("image_comparaison/comparaison_non_relativistic.png", dpi=300)        
    plt.show()

def plot_3component(L_t, L_r, save=False):
    """
    Plots the three components (x, y, z) of the trajectory as a function of time.
    Args:
        L_t: Array of time steps
        L_r: Array of positions
        save: If True, saves the plot as a PNG file
    """
    fig = plt.figure(num="3 Component", figsize=(12, 5))
    fig.suptitle("All 3 components")
    gs = gridspec.GridSpec(1, 3, fig)

    ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])

    ax1.plot(L_t, L_r[:, 0], label="X-component")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Displacement [m]")

    ax2.plot(L_t, L_r[:, 1], label="Y-Component")
    ax2.set_xlabel("Time [s]")

    ax3.plot(L_t, L_r[:, 2], label="Z-Component")
    ax3.set_xlabel("Time [s]")

    plt.tight_layout()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    if save:
        plt.savefig("all_component.png", dpi=300)

def plot_2D(L_x, L_y, save=False):
    """
    Plots the 2D trajectory in the x-y plane.
    Args:
        L_x: Array of x positions
        L_y: Array of y positions
        save: If True, saves the plot as a PNG file
    """
    plt.figure(num="Plot 2D", figsize=(12, 5))
    step = 1
    plt.plot(L_x[step::], L_y[step::], "--", label="Displacement in time")
    plt.xlabel("X-component [m]")
    plt.ylabel("Y-component [m]")
    plt.title("Plane trajectory")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig("2D_plot.png", dpi=300)
    plt.show()

def plot_energy(L_t, L_v, m, c = const.c, relativistic = False, save=False):
    """
    Plots the kinetic energy as a function of time.
    Args:
        L_t: Array of time steps
        L_v2: Array of velocity magnitudes
        m: Particle mass
        save: If True, saves the plot as a PNG file
    """
    if relativistic:
        plt.figure("Energy for a relativistic particle", figsize=(12, 5))
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.plot(L_t, 1 / np.sqrt(1 - (np.linalg.norm(L_v, axis=1)/c)**2) * m * c**2, label="Kinetic Energy")
        
        plt.tight_layout()
        if save:
            plt.savefig("image_comparaison/energy_relativistic.png", dpi=300)  # CORRECTION
    else:
        plt.figure("Energy for a non-relativistic particle", figsize=(12, 5))
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.plot(L_t, 1 / np.sqrt(1 - (np.linalg.norm(L_v, axis=1)/c)**2) * m * c**2, label="Kinetic Energy")

        plt.tight_layout()
        if save:
            plt.savefig("image_comparaison/energy_non_relativistic.png", dpi=300)  # CORRECTION

    plt.show()

#------------------------------------ Relativistic Electron ----------------------------------#

# ---------- Relativistic Boris with Leapfrog using classical gamma expression ----------
def RelativisticBorisPusher(r0, v0, E, B, dt, q, m, c=const.c):
    gamma_0 = 1.0 / np.sqrt(1.0 - (np.linalg.norm(v0)/c)**2)
    L_u = [v0 * gamma_0]
    L_v = [v0]
    L_r = [r0]
    for i in range(1, len(L_t)):  # Changé de np.shape(L_t)[0] à len(L_t)
        L_r.append(L_r[i-1] + L_v[i-1] * dt)
        
        u_onehalf = L_u[i-1] + q * dt / (2 * m) * (E + np.cross(L_v[i-1], B))

        u_prime = u_onehalf + q * dt / (2 * m) * E
        gamma_prime = np.sqrt(1.0 + (np.linalg.norm(u_prime)/c)**2)
        
        tau = q * dt / (2 * m) * B
        u_star = np.dot(u_prime, tau) / c

        sigma = gamma_prime**2 - np.dot(tau, tau)
        
        gamma_i = np.sqrt((sigma + np.sqrt(sigma**2 + 4 * (np.dot(tau,tau) + np.dot(u_star,u_star)))) / 2)
        t = tau / gamma_i
        s = 1 / (1 + np.dot(t, t))
        u_i = s * (u_prime + np.dot(u_prime, t) * t + np.cross(u_prime, t))
        v_i = u_i / gamma_i
        L_u.append(u_i)
        L_v.append(v_i)
    
    return np.array(L_r), np.array(L_v)

def non_relativistic_analytical_solution(L_t, r0, v0, q, m, B, E):
    """ 
    Solution analytique pour la vitesse d'une particule chargée non relativiste dans des champs E et B constants (Bz uniquement) en prenant en compte la dérive électrique.
    Retourne un array de vitesses à chaque pas de temps."""

    v_drift = np.cross(E, B) / np.linalg.norm(B)**2
    omega = np.abs(q) * np.linalg.norm(B) / m
    v_analytical = []
    r_analytical = [r0]
    for t in L_t:
        vx_t = (v0[0] - v_drift[0]) * np.cos(omega*t) + np.sign(q) * (v0[1] - v_drift[1]) * np.sin(omega*t) + v_drift[0]
        vy_t = (v0[1] - v_drift[1]) * np.cos(omega*t) - np.sign(q) * (v0[0] - v_drift[0]) * np.sin(omega*t) + v_drift[1]
        vz_t = v0[2] + q * E[2] / m * t
        v_analytical.append(np.array([vx_t, vy_t, vz_t]))
    v_analytical = np.array(v_analytical)
    for i in range(1,np.shape(L_t)[0]):
        r_analytical.append(r_analytical[i-1] + v_analytical[i-1]*dt)
    r_analytical = np.array(r_analytical)
    return r_analytical

def relativistic_analytical_solution(L_t, r0, v0, q, m, B, E, c=const.c):
    """
    Solution analytique pour la vitesse d'une particule chargée relativiste dans des champs E et B constants (Bz uniquement).
    Retourne un array de vitesses à chaque pas de temps.
    """
    u = np.cross(E, B) / np.linalg.norm(E)**2 * c**2
    E_prime = E * np.sqrt((np.dot(E,E) - np.dot(B,B)*c**2)/np.dot(E,E))

    gamma = 1.0 / np.sqrt(1.0 - (np.linalg.norm(u)/c)**2)
    L_tprime = [L_t[0]]
    for i in range(1, np.shape(L_t)[0]):
        L_tprime.append((L_t[i] - L_t[i-1]) / gamma + L_tprime[i-1]) # Transformation de Lorentz du temps
    L_tprime = np.array(L_tprime)

    v0_prime = np.zeros_like(v0)
    v0_prime[0] = v0[0] / gamma / (1 - np.dot(v0, u)/c**2)
    v0_prime[1] = (v0[1] - u[1]) / (1 - np.dot(v0, u)/c**2)  # Transformation de Lorentz de la vitesse
    v0_prime[2] = v0[2] / gamma / (1 - np.dot(v0, u)/c**2)
    gamma0_prime = 1.0 / np.sqrt(1.0 - (np.linalg.norm(v0_prime)/c)**2)
    p_prime = [m * v0_prime * gamma0_prime]
    for i,t in enumerate(L_tprime[1:]):
        p_prime.append(q * E_prime * t + p_prime[0])  # Impulsion dans le référentiel primé
    p_prime = np.array(p_prime)

    gamma_prime = np.sqrt(1.0 + (np.linalg.norm(p_prime, axis=1)/(m*c))**2)
    v_prime = p_prime / (m * gamma_prime[:, np.newaxis])  # Vitesse dans le référentiel primé
    v = np.zeros_like(v_prime)
    v[:,0] = v_prime[:,0] / gamma / (1 + np.dot(v_prime, u)/c**2)
    v[:,1] = (v_prime[:,1] + u[1]) / (1 + np.dot(v_prime, u)/c**2)  # Transformation de Lorentz inverse de la vitesse
    v[:,2] = v_prime[:,2] / gamma / (1 + np.dot(v_prime, u)/c**2)

    r_analytical = []
    for i in range(len(L_t)):
        x = r0[0] + v[i,0]*L_t[i]
        y = r0[1] + v[i,1]*L_t[i]
        z = r0[2] + v[i,2]*L_t[i]
        r_analytical.append(np.array([x, y, z]))
    r_analytical = np.array(r_analytical)
    return r_analytical, v

#Initial conditions and simulation parameters for non-relativistic case
t0, tf, dt = 0, 1e-9, 1e-14
r0 = np.array([1, 1, 1])
v0 = np.array([1e3, 1e3, 0.0])  # Initial velocity corresponding to low energy (1 keV)
q, m = -const.e, const.m_e
E_field = np.array([0, 0.0, 0.0])  # No electric field
B_field = np.array([0.0, 0.0, 0.5*1e-6])    # Magnetic field in z direction 

# Run Non-Relativistic Boris pusher with leapfrog integration
L_t = np.arange(t0, tf, dt)

L_r, L_v = RelativisticBorisPusher(r0, v0, E_field, B_field, dt, q, m)
L_r_analytic = non_relativistic_analytical_solution(L_t, r0, v0, q, m, B_field, E_field)
# Plot results for non-relativistic case
comparaison(L_t, L_r, L_r_analytic, False, True)
plot_energy(L_t, L_v,m,const.c,False,True)

# Initial conditions and simulation parameters for relativistic case
t0, tf, dt = 0, 1e-9, 1e-14
r0 = np.array([1, 1, 1])
v0 = np.array([0.90 * const.c / np.sqrt(2), 0.90 * const.c / np.sqrt(2), 0.0])  # Initial velocity corresponding to high energy (90% speed of light)
q, m, light_speed = -const.e, const.m_e, const.c
E_field = np.array([1e11, 0.0, 0.0])  # No electric field
B_field = np.array([0.0, 0.0, 0.5*1e2])    # Magnetic field in z direction 

# Run Relativistic Boris pusher with leapfrog integration
L_t = np.arange(t0, tf, dt)

L_r, L_v = RelativisticBorisPusher(r0, v0, E_field, B_field, dt, q, m, light_speed)
L_r_analytic, v_analytic = relativistic_analytical_solution(L_t, r0, v0, q, m, B_field, E_field, light_speed)

# Plot results for relativistic case
comparaison(L_t, L_r, L_r_analytic,True, True)
# plot_2D(L_r[:, 0], L_r[:, 1])
plot_energy(L_t, L_v, m, light_speed, True, True)
