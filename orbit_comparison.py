import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.constants as const

from math import sqrt, asinh

c = cst.c  # speed of light (m/s)

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
def RelativisticBorisPusher(t_array,r0, v0, E, B, dt, q, m, c=const.c):
    gamma_0 = 1.0 / np.sqrt(1.0 - (np.linalg.norm(v0)/c)**2)
    L_u = [v0 * gamma_0]
    L_v = [v0]
    L_r = [r0]
    for i in range(1, len(t_array)):  # Changé de np.shape(L_t)[0] à len(L_t)
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

def particle_position_lab(t_lab_array, m, q, E_mag, B_mag, v0_y,
                          tol=1e-9, max_bisect_iter=80):
    """
    Compute (x,y,z) positions in the lab frame for a charged relativistic particle
    in uniform fields E=(E_mag,0,0) and B=(0,0,B_mag), initial lab velocity v0=(0,v0_y,0).
    The algorithm:
      1. compute drift velocity u = c * (E x B) / E^2  (here u is along ±y)
      2. boost to K'' moving with u, solve motion there (analytic integrals)
      3. inverse-Lorentz transform back; for each lab time t_lab solve implicitly
         for t' using bisection on t_lab - gamma_u*(t' + u*y'(t')/c^2) = 0

    Parameters
    ----------
    t_lab_array : array_like
        Lab times (s) at which to compute the particle position (must be >= 0).
    m : float
        Particle rest mass (kg).
    q : float
        Particle charge (C).
    E_mag : float
        Magnitude of E-field (V/m) along +x.
    B_mag : float
        Magnitude of B-field (T) along +z.
    v0_y : float
        Initial lab y-velocity (m/s).
    tol : float
        Tolerance for bisection in seconds.
    max_bisect_iter : int
        Max iterations for bisection.

    Returns
    -------
    positions : ndarray, shape (N,3)
        x,y,z positions in lab frame for each requested lab time.
    times_out : ndarray
        The input t_lab_array as a numpy array (sorted to match outputs).

    Notes
    -----
    - The code assumes E_mag > B_mag for a physical drift speed u < c.
    - Units must be SI.
    """
    t_lab_array = np.asarray(t_lab_array, dtype=float)
    if np.any(t_lab_array < 0):
        raise ValueError("All requested lab times must be non-negative.")
    if E_mag == 0.0:
        raise ValueError("E_mag must be nonzero for the transformation used here.")
    # Drift velocity (vector): u = c * (E x B) / E^2
    # for E=(E,0,0), B=(0,0,B) => E x B = (0, -E*B, 0) => u = (0, -c*B/E, 0)
    u_y = -c * (B_mag / E_mag)
    u = np.array([0.0, u_y, 0.0])
    u_mag = abs(u_y)
    if u_mag >= c:
        raise ValueError("Drift speed |u| >= c (requires E_mag > B_mag).")

    gamma_u = 1.0 / sqrt(1.0 - (u_mag / c)**2)

    # Transform initial particle velocity v0 (lab) to K'' (boost along y)
    # velocity-transform: v'_|| = (v|| - u) / (1 - v·u / c^2)
    # here v0 = (0, v0_y, 0) and u = (0, u_y, 0)
    denom = 1.0 - (v0_y * u_y) / (c**2)
    v_y_prime_0 = (v0_y - u_y) / denom
    # initial v_x' = 0 because v_x_lab = 0 and perpendicular velocity transform factor multiplies zero
    v_x_prime_0 = 0.0
    # initial gamma in K''
    gamma0_prime = 1.0 / sqrt(1.0 - (v_y_prime_0**2 + v_x_prime_0**2) / c**2)
    # conserved transverse momentum in K'': p_y = m * gamma0' * v_y'
    p_y = m * gamma0_prime * v_y_prime_0

    # Electric field in K'': E' = E / gamma_u  (from the document)
    E_prime = E_mag / gamma_u
    qEprime = q * E_prime

    # helper dimensionless constant for compact formulas
    def P_const():
        return p_y / (m * c)  # dimensionless p_y/(m c)

    P = P_const()

    # x'(t') and y'(t') in K'', derived analytically (starting at origin at t'=0)
    # define S(t') = p_x/(m c) = (q E' t')/(m c)
    def S_of_tprime(tprime):
        return (qEprime * tprime) / (m * c)

    def gamma_total_from_S(S):
        return sqrt(1.0 + P**2 + S**2)

    def x_prime_of_tprime(tprime):
        S = S_of_tprime(tprime)
        pref = (m * c**2) / qEprime
        return pref * (gamma_total_from_S(S) - sqrt(1.0 + P**2))

    def y_prime_of_tprime(tprime):
        S = S_of_tprime(tprime)
        # y' = (p_y c / qE') * asinh( S / sqrt(1+P^2) )
        pref = (p_y * c) / qEprime
        denom_sqrt = sqrt(1.0 + P**2)
        return pref * asinh(S / denom_sqrt)

    # inverse transform to lab from K'':
    # boost inverse (K'' -> lab) for boost velocity u along +y:
    # x_lab = x'
    # y_lab = gamma_u * (y' + u * t')
    # t_lab = gamma_u * (t' + u * y' / c^2)
    def lab_coords_from_tprime(tprime):
        x_p = x_prime_of_tprime(tprime)
        y_p = y_prime_of_tprime(tprime)
        x_lab = x_p
        y_lab = gamma_u * (y_p + u_y * tprime)
        # t_lab returned for root-finding
        t_lab_calc = gamma_u * (tprime + (u_y * y_p) / (c**2))
        return x_lab, y_lab, t_lab_calc

    # For each desired t_lab, find t' s.t. t_lab_calc(t') = t_lab via bisection
    def find_tprime_for_tlab(tlab):
        # f(t') = t_lab_calc(t') - tlab; f(0) = -tlab (since t_lab_calc(0)=0)
        f0 = -tlab
        if abs(f0) < tol:
            return 0.0
        # find an upper bound t_hi so that f(t_hi) > 0
        t_hi = max(1e-12, tlab / max(1e-12, gamma_u))  # crude initial guess
        # ensure positive f at t_hi by expanding until sign change
        f_hi = lab_coords_from_tprime(t_hi)[2] - tlab
        n_expand = 0
        while f_hi <= 0 and n_expand < 200:
            t_hi *= 2.0
            f_hi = lab_coords_from_tprime(t_hi)[2] - tlab
            n_expand += 1
        if f_hi <= 0:
            raise RuntimeError("Failed to bracket root for t_lab = {} (try larger max expansion)".format(tlab))

        # bisection
        t_lo = 0.0
        f_lo = f0
        iter_count = 0
        while iter_count < max_bisect_iter:
            t_mid = 0.5 * (t_lo + t_hi)
            f_mid = lab_coords_from_tprime(t_mid)[2] - tlab
            if abs(f_mid) < tol:
                return t_mid
            # choose half-interval
            if f_mid > 0:
                t_hi, f_hi = t_mid, f_mid
            else:
                t_lo, f_lo = t_mid, f_mid
            iter_count += 1
        # last resort return midpoint
        return 0.5 * (t_lo + t_hi)

    # compute positions for every requested lab time
    positions = np.zeros((t_lab_array.size, 3), dtype=float)
    for i, tl in enumerate(t_lab_array):
        tprime = find_tprime_for_tlab(tl)
        x_lab, y_lab, _ = lab_coords_from_tprime(tprime)
        positions[i, 0] = x_lab
        positions[i, 1] = y_lab
        positions[i, 2] = 0.0

    return positions, t_lab_array
# Initial conditions and simulation parameters for relativistic case
t0, tf, dt = 0, 1e-9, 1e-14
r0 = np.array([1, 1, 1])
v0 = np.array([0.90 * const.c / np.sqrt(2), 0.90 * const.c / np.sqrt(2), 0.0])  # Initial velocity corresponding to high energy (90% speed of light)
q, m, light_speed = -const.e, const.m_e, const.c
E_field = np.array([1e11, 0.0, 0.0])  # No electric field
B_field = np.array([0.0, 0.0, 0.5*1e2])    # Magnetic field in z direction 

# Run Relativistic Boris pusher with leapfrog integration
L_t = np.arange(t0, tf, dt)

L_r, L_v = RelativisticBorisPusher(L_t,r0, v0, E_field, B_field, dt, q, m, light_speed)
L_r_analytic, v_analytic = particle_position_lab(L_t, m, q, E_field, B_field, v0,
                          tol=1e-9, max_bisect_iter=80)

# Plot results for relativistic case
comparaison(L_t, L_r, L_r_analytic,True, True)
# plot_2D(L_r[:, 0], L_r[:, 1])
plot_energy(L_t, L_v, m, light_speed, True, True)
