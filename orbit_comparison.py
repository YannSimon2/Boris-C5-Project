import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.constants as const

#------------------------------------ Plotting functions --------------------------------#

def comparaison(L_t, L_r, L_analytic, save=False):
    """
    Plots the numerical and analytical trajectories and their relative errors for each axis.
    Args:
        L_t: Array of time steps
        L_r: Numerical positions
        L_analytic: Analytical positions
        save: If True, saves the plot as a PNG file
    """
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(2, 3, fig, height_ratios=[2, 1], hspace=0)
    fig.suptitle("Trajectory and error on each axis")

    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax3, ax4 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    ax5, ax6 = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])

    # X component
    ax1.plot(L_t, L_r[:, 0], label="Numerical", color="blue")
    ax1.plot(L_t, L_analytic[:, 0], "--", label="Analytic", color="red")
    ax1.set_xticks([])
    ax1.set_ylabel("Displacement")
    ax2.plot(L_t, np.abs((L_r[:, 0] - L_analytic[:, 0]) / L_analytic[:, 0]) * 100, label="Relative error", color="green")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Error [%]")

    # Y component
    ax3.plot(L_t, L_r[:, 1], label="Numerical", color="blue")
    ax3.plot(L_t, L_analytic[:, 1], "--", label="Analytic", color="red")
    ax3.set_xticks([])
    ax4.plot(L_t, np.abs((L_r[:, 1] - L_analytic[:, 1]) / L_analytic[:, 1]) * 100, label="Relative error", color="green")
    ax4.set_xlabel("Time [s]")

    # Z component
    ax5.plot(L_t, L_r[:, 2], label="Numerical", color="blue")
    ax5.plot(L_t, L_analytic[:, 2], "--", label="Analytic", color="red")
    ax5.set_xticks([])
    ax5.legend()
    ax6.plot(L_t, np.abs((L_r[:, 2] - L_analytic[:, 2]) / L_analytic[:, 2]) * 100, label="Relative error", color="green")
    ax6.set_xlabel("Time [s]")

    plt.tight_layout()
    if save:
        plt.savefig("comparaison.png", dpi=300)
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

def plot_energy(L_t, L_v2, m, relativistic = False, save=False):
    """
    Plots the kinetic energy as a function of time.
    Args:
        L_t: Array of time steps
        L_v2: Array of velocity magnitudes
        m: Particle mass
        save: If True, saves the plot as a PNG file
    """

    plt.figure("Energy", figsize=(12, 5))
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    if relativistic:
        plt.title(r"Energy conservation ($\vec{E}=0$ and relativistic)")
        # Relativistic energy: E = (gamma - 1) * m * c²
        # where gamma = 1 / sqrt(1 - (v/c)²)
        gamma = 1.0 / np.sqrt(1.0 - (L_v2 / const.c)**2)
        L_E = (gamma - 1) * m * const.c**2
    else:
        plt.title(r"Energy conservation ($\vec{E}=0$)")
        L_E = 0.5 * m * L_v2**2
    plt.plot(L_t, L_E)
    plt.tight_layout()

    if save:
        plt.savefig("energy.png", dpi=300)

    plt.show()

#------------------------------------ Relativistic Electron ----------------------------------#

# ---------- Relativistic Boris with Leapfrog using classical gamma expression ----------
def RelativisticBorisPusher(v_n_minus_half, E, B, dt, q, m, c=const.c):
    """
    Relativistic Boris algorithm with leapfrog integration for a single charged particle.
    Uses the classical expression: gamma = 1 / sqrt(1 - (v/c)²)
    Args:
        v_n_minus_half: Velocity at time n-1/2 (3D numpy array)
        E: Electric field vector (3D numpy array) 
        B: Magnetic field vector (3D numpy array)
        dt: Time step
        q: Particle charge
        m: Particle mass
        c: Speed of light
    Returns:
        v_n_plus_half: Updated velocity at time n+1/2
    """
    qm = q / m
    
    # First half-step acceleration by electric field
    v_minus = v_n_minus_half + qm * E * (dt / 2.0)
    
    # Calculate relativistic factor using classical expression
    v_speed = np.linalg.norm(v_minus)
    gamma_minus = 1.0 / np.sqrt(1.0 - (v_speed / c)**2)
    
    # Magnetic rotation step
    t = (q * B) * (dt / (2.0 * m * gamma_minus))
    t_magnitude_squared = np.dot(t, t)
    s = 2.0 * t / (1.0 + t_magnitude_squared)
    
    # Boris rotation
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)
    
    # Second half-step acceleration by electric field
    v_n_plus_half = v_plus + qm * E * (dt / 2.0)
    
    return v_n_plus_half

def relativistic_analytical_solution(L_t, r0, v0, q, m, B, E, c=const.c):
    """
    Solution analytique pour la vitesse d'une particule chargée relativiste dans des champs E et B constants (Bz uniquement).
    Retourne un array de vitesses à chaque pas de temps.
    """
    if np.allclose(E, np.array([0.0, 0.0, 0.0])):
        # Cas sans champ électrique : mouvement cyclotron relativiste
        v0_speed = np.linalg.norm(v0)
        gamma0 = 1.0 / np.sqrt(1.0 - (v0_speed / c)**2)
        omega = np.abs(q) * np.linalg.norm(B) / (gamma0 * m)
        v_analytical = []
        r_analytical = [r0]
        for t in L_t:
            vx_t = v0[0] * np.cos(omega*t) - v0[1] * np.sin(omega*t)
            vy_t = v0[1] * np.cos(omega*t) + v0[0] * np.sin(omega*t)
            vz_t = v0[2]
            v_analytical.append(np.array([vx_t, vy_t, vz_t]))
        v_analytical = np.array(v_analytical)
        for i in range(1,np.shape(L_t)[0]):
            r_analytical.append(r_analytical[i-1] + v_analytical[i-1]*dt)
        r_analytical = np.array(r_analytical)
    else:
        # Cas croisé E et B relativiste dans le cas où E est selon x et B selon z
        if np.linalg.norm(E) <= c * np.linalg.norm(B):
            u = np.cross(E, B) / np.linalg.norm(B)**2
            E_prime = np.array([0, 0, 0])
            B_prime = B * np.sqrt(1 - (np.linalg.norm(u)/c)**2)

        else:
            u = np.cross(E, B) / np.linalg.norm(E)**2 * c**2
            E_prime = E * np.sqrt(1 - (np.linalg.norm(u)/c)**2)
            B_prime = np.array([0, 0, 0])

        gamma = 1.0 / np.sqrt(1.0 - (np.linalg.norm(u)/c)**2)
        omega = np.abs(q) * np.linalg.norm(B_prime) / (gamma * m)
        L_tprime = [L_t[0]]
        for i in range(1, np.shape(L_t)[0]):
            L_tprime.append(gamma * (L_t[i] - L_t[i-1])+L_tprime[i-1]) # Transformation de Lorentz du temps
        L_tprime = np.array(L_tprime)
        # Dans le référentiel primé, le mouvement est un cyclotron classique avec les champs primés et un drift en plus
        v_analytical_prime = []
        r_analytical_prime = [r0]
        for i, t in enumerate(L_tprime):
            vx_t = (v0[0] - u[0]) * np.cos(omega*t) - (v0[1] - u[1]) * np.sin(omega*t) + u[0]
            vy_t = (v0[1] - u[1]) * np.cos(omega*t) + (v0[0] - u[0]) * np.sin(omega*t) + u[1]
            vz_t = v0[2] + q * E_prime[2] / m * t
            v_analytical_prime.append(np.array([vx_t, vy_t, vz_t]))
            if i > 0:
                r_analytical_prime.append(r_analytical_prime[i-1] + v_analytical_prime[i-1]*dt)
        # Transformation inverse de Lorentz pour obtenir les positions dans le référentiel initial
        r_analytical_prime = np.array(r_analytical_prime)
        r_analytical = []
        for i in range(np.shape(r_analytical_prime)[0]):
            x = gamma * (r_analytical_prime[i][0] + u[0]*L_tprime[i])
            y = gamma * (r_analytical_prime[i][1] + u[1]*L_tprime[i])
            z = r_analytical_prime[i][2]
            r_analytical.append(np.array([x, y, z]))
        r_analytical = np.array(r_analytical)

    return r_analytical

# Initial conditions and simulation parameters for relativistic case
t0, tf, dt = 0, 1e-6, 1e-10
r0 = np.array([1e-6, 1e-6, 1e-6])
v0 = np.array([1e-6 * const.c, 0.0, 0.0])  # Initial velocity corresponding to high energy (90% speed of light)
q, m = -const.elementary_charge, const.electron_mass
E_field = np.array([1e-4, 0.0, 0.0])  # No electric field
B_field = np.array([0.0, 0.0, 1e-6])    # Magnetic field in z direction

# Run Relativistic Boris pusher with leapfrog integration
L_t = np.arange(t0, tf, dt)
L_v = np.zeros((np.shape(L_t)[0], 3))
L_r = np.zeros((np.shape(L_t)[0], 3))
L_r[0, :] = r0
L_v[0, :] = v0

# Initialize velocity at t = -dt/2 for leapfrog scheme
# First step: advance velocity from t=0 to t=dt/2
v_half = v0 + (q/m) * E_field * (dt/2.0)

for n in range(1, np.shape(L_t)[0]):
    # Update position using velocity at n-1/2
    L_r[n, :] = L_r[n-1, :] + v_half * dt
    
    # Update velocity from n-1/2 to n+1/2 using Boris algorithm
    v_half = RelativisticBorisPusher(v_half, E_field, B_field, dt, q, m, const.c)
    
    # Store velocity at integer time step (interpolated)
    L_v[n, :] = v_half
    L_r[n, :] = L_r[n-1, :] + v_half * dt

L_r_analytic, L_v_analytic = relativistic_analytical_solution(L_t, r0, v0, q, m, B_field, E_field, const.c)

# Plot results for relativistic case
comparaison(L_t, L_r, L_r_analytic)
# plot_2D(L_r[:, 0], L_r[:, 1])
plot_energy(L_t, np.linalg.norm(L_v, axis=1), m, relativistic=True)
