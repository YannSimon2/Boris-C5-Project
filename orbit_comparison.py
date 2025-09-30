import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.constants as const

#------------------------------------ Boris pusher Implementation -----------------------------------------#

def Boris_pusher(r0, v0, t0, tf, dt, E, B, q, m):
    """
    Implements the Boris algorithm to advance the position and velocity of a charged particle
    in given electric and magnetic fields.
    Args:
        r0: Initial position (numpy array)
        v0: Initial velocity (numpy array)
        t0: Initial time
        tf: Final time
        dt: Time step
        E: Function returning electric field
        B: Function returning magnetic field
        q: Particle charge
        m: Particle mass
    Returns:
        L_t: Array of time steps
        L_r: Array of positions at each time step
        L_v: Array of velocities at each time step
    """
    L_t = np.arange(t0, tf, dt)
    L_r = np.zeros((np.shape(L_t)[0], 3))
    L_r[0, :] = r0
    L_v = np.zeros((np.shape(L_t)[0], 3))
    L_v[0, :] = v0

    for n in range(1, np.shape(L_t)[0]):
        rn, vn, tn = L_r[n-1, :], L_v[n-1, :], L_t[n-1]
        # Half acceleration by E
        vminus = vn + q / m * E * dt / 2

        # Rotation due to B
        t = q * dt / (2 * m) * B
        s = 2 / (1 + np.dot(t, t)) * t

        vprime = vminus + np.cross(vminus, t)
        vplus = vminus + np.cross(vprime, s)

        # Second half acceleration by E
        vnp1 = vplus + q / m * E * dt / 2
        rnp1 = rn + vn * dt
        L_r[n, :] = rnp1
        L_v[n, :] = vnp1

    return L_t, L_r, L_v

def analytical_solution(L_t, r0, v0, q, m, B, E):
    """
    Analytical solution for the trajectory of a charged particle in uniform, constant, crossed E and B fields (Bz only).
    Args:
        L_t: Array of time steps
        r0: Initial position
        v0: Initial velocity
        q: Particle charge
        m: Particle mass
        B: Magnetic field vector
        E: Electric field vector
    Returns:
        r_analytical: Array of analytical positions at each time step
    """
    omega = np.abs(q) * np.linalg.norm(B) / m
    E_cross_B = np.cross(E, B) / np.linalg.norm(B)**2
    r_analytical = []
    for t in L_t:
        x_t = r0[0] + (v0[0] - E_cross_B[0]) * np.sin(omega*t) / omega - (v0[1] - E_cross_B[1]) * (1-np.cos(omega*t)) / omega + E_cross_B[0] * t
        y_t = r0[1] + (v0[1] - E_cross_B[1]) * np.sin(omega*t) / omega + (v0[0] - E_cross_B[0]) * (1-np.cos(omega*t)) / omega + E_cross_B[1] * t
        z_t = r0[2] + v0[2] * t + q * E[2] / (2 * m) * t**2
        r_analytical.append(np.array([x_t, y_t, z_t]))
    return np.array(r_analytical)

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
    step = 100
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

#------------------------------------ Main execution ----------------------------------#

# Physical constants for electron
q, m = -const.elementary_charge, const.electron_mass
E, B = np.array([1e-6, 0, 0]), np.array([0, 0, 1e-9])

# Initial conditions and simulation parameters
t0, tf, dt = 0, 1e-8, 1e-11
r0 = np.array([0.1, 0.1, 0.1])
v0 = np.array([1e-2, 0, 0])

# Run Boris pusher and analytical solution
# L_t, L_r, L_v = Boris_pusher(r0, v0, t0, tf, dt, E, B, q, m)
# L_analytic = analytical_solution(L_t, r0, v0, q, m, B, E)

# Plot results
# comparaison(L_t, L_r, L_analytic)
# plot_3component(L_t, L_r)
# plot_2D(L_r[:, 0], L_r[:, 1])
# plot_energy(L_t, np.linalg.norm(L_v, axis=1), m)

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

def velocity_to_momentum(v, m, c=const.c):
    """
    Convert velocity to momentum for a single relativistic particle.
    Uses the classical expression: gamma = 1 / sqrt(1 - (v/c)²)
    Args:
        v: Velocity vector (3D numpy array)
        m: Particle mass
        c: Speed of light
    Returns:
        p: Momentum vector
    """
    v_speed = np.linalg.norm(v)
    gamma = 1.0 / np.sqrt(1.0 - (v_speed / c)**2)
    p = gamma * m * v
    return p

def momentum_to_velocity(p, m, c=const.c):
    """
    Convert momentum to velocity for a single relativistic particle.
    Uses the classical expression: gamma = 1 / sqrt(1 - (v/c)²)
    Args:
        p: Momentum vector (3D numpy array)
        m: Particle mass
        c: Speed of light
    Returns:
        v: Velocity vector
    """
    # For relativistic particles: p = gamma * m * v
    # We need to solve: |p| = gamma * m * |v| where gamma = 1/sqrt(1-(v/c)²)
    # This gives: |v| = |p| / (m * sqrt(1 + (|p|/(mc))²))
    p_magnitude = np.linalg.norm(p)
    v_magnitude = p_magnitude / (m * np.sqrt(1.0 + (p_magnitude / (m * c))**2))
    
    # Direction is the same as momentum
    if p_magnitude > 0:
        v = (v_magnitude / p_magnitude) * p
    else:
        v = np.zeros(3)
    
    return v

def relativistic_analytical_solution(L_t, r0, v0, q, m, B, E, c=const.c):
    """
    Analytical solution for the trajectory of a relativistic charged particle in uniform, constant, crossed E and B fields (Bz only).
    Uses the classical expression: gamma = 1 / sqrt(1 - (v/c)²)
    Args:
        L_t: Array of time steps
        r0: Initial position
        v0: Initial velocity
        q: Particle charge
        m: Particle mass
        B: Magnetic field vector
        E: Electric field vector
        c: Speed of light
    Returns:
        r_analytical: Array of analytical positions at each time step
    """
    if np.allclose(E, np.array([0.0, 0.0, 0.0])):
        # Calculate relativistic cyclotron frequency
        v0_speed = np.linalg.norm(v0)
        gamma0 = 1.0 / np.sqrt(1.0 - (v0_speed / c)**2)
        omega = np.abs(q) * np.linalg.norm(B) / (gamma0 * m)  # Relativistic cyclotron frequency
        
        r_analytical = []
        for t in L_t:
            x_t = r0[0] + (v0[0]) * np.sin(omega*t) / omega - (v0[1]) * (1-np.cos(omega*t)) / omega
            y_t = r0[1] + (v0[1]) * np.sin(omega*t) / omega + (v0[0]) * (1-np.cos(omega*t)) / omega
            z_t = r0[2] + v0[2] * t
            r_analytical.append(np.array([x_t, y_t, z_t]))
    else:
        # For crossed E and B fields with relativistic effects
        if np.linalg.norm(E) <= c * np.linalg.norm(B):
            u = np.cross(E, B) / np.linalg.norm(B)**2
            E_prime = np.array([0, 0, 0])
            B_prime = B * np.sqrt(1 - (np.linalg.norm(u)/c)**2)
        else:
            u = np.cross(E, B) / np.linalg.norm(E)**2 * c**2
            E_prime = E * np.sqrt(1 - (np.linalg.norm(u)/c)**2)
            B_prime = np.array([0, 0, 0])
        
        v0_speed = np.linalg.norm(v0)
        gamma0 = 1.0 / np.sqrt(1.0 - (v0_speed / c)**2)
        omega = np.abs(q) * np.linalg.norm(B_prime) / (gamma0 * m)
        
        r_analytical = []
        for t in L_t:
            x_t = r0[0] + (v0[0] - u[0]) * np.sin(omega*t) / omega - (v0[1] - u[1]) * (1-np.cos(omega*t)) / omega + u[0] * t
            y_t = r0[1] + (v0[1] - u[1]) * np.sin(omega*t) / omega + (v0[0] - u[0]) * (1-np.cos(omega*t)) / omega + u[1] * t
            z_t = r0[2] + v0[2] * t + q * E_prime[2] / (2 * gamma0 * m) * t**2
            r_analytical.append(np.array([x_t, y_t, z_t]))
        # Transform back to original frame
        r_analytical = [np.array([x + u[0]*t, y + u[1]*t, z + u[2]*t]) for (x,y,z), t in zip(r_analytical, L_t)]
    
    return np.array(r_analytical)

# Initial conditions and simulation parameters for relativistic case
t0, tf, dt = 0, 1e-8, 1e-11
r0 = np.array([0.1, 0.1, 0.1])
v0 = np.array([0.0, 0.0, 0.9 * const.c])  # Initial velocity corresponding to high energy (90% speed of light)
q, m = -const.elementary_charge, const.electron_mass
E_field = np.array([1e-6, 0.0, 0.0])  # No electric field
B_field = np.array([0.0, 0.0, 1e-8])    # Magnetic field in z direction

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

L_analytic = relativistic_analytical_solution(L_t, r0, v0, q, m, B_field, E_field, const.c)

# Plot results for relativistic case
comparaison(L_t, L_r, L_analytic)
plot_energy(L_t, np.linalg.norm(L_v, axis=1), m, relativistic=True)
