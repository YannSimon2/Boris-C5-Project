import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.constants as const

#################################### Implémentation Boris pusher ##########################################

def Electric_field(r,t):
    return np.array([1e-6,0,0]) #E field 1e-6

def Magnetic_field(r,t):
    return np.array([0,0,1e-9]) #B field 1e-8,-9

def Boris_pusher(r0, v0, t0, tf, dt, E, B, q, m):
    L_t = np.arange(t0,tf,dt)
    L_r = np.zeros((np.shape(L_t)[0],3))
    L_r[0,:] = r0
    L_v = np.zeros((np.shape(L_t)[0],3))
    L_v[0,:] = v0

    for n in range(1,np.shape(L_t)[0]):
        rn, vn, tn = L_r[n-1,:], L_v[n-1,:], L_t[n-1]
        vminus = vn + q / m * E(rn, tn + dt/2) * dt / 2

        t = q * dt / (2 * m) * B(rn, tn + dt/2)
        s = 2 / (1 + np.dot(t,t)) * t
        
        vprime = vminus + np.cross(vminus, t)
        vplus = vminus + np.cross(vprime, s)

        vnp1 = vplus + q / m * E(rn, tn + dt/2) * dt / 2
        rnp1 = rn + vn * dt
        L_r[n,:] = rnp1
        L_v[n,:] = vnp1
    
    return L_t, L_r, L_v

def analytical_solution(L_t, r0, v0, q, m, B, E): # Solution for a drift in crossed E and B fields (uniform and constant) only for Bz
    omega = np.abs(q) * np.linalg.norm(B) / m
    E_cross_B = np.cross(E, B) / np.linalg.norm(B)**2
    r_analytical = []
    for t in L_t:
        x_t = r0[0] + (v0[0] - E_cross_B[0]) * np.sin(omega*t) / omega - (v0[1] - E_cross_B[1]) * (1-np.cos(omega*t)) / omega + E_cross_B[0] * t
        y_t = r0[1] + (v0[1] - E_cross_B[1]) * np.sin(omega*t) / omega + (v0[0] - E_cross_B[0]) * (1-np.cos(omega*t)) / omega + E_cross_B[1] * t
        z_t = r0[2] + v0[2] * t + q * E[2] / (2 * m) * t**2
        r_analytical.append(np.array([x_t,y_t,z_t]))
        # r_analytical.append(r0 + (v0 - E_cross_B) / omega * np.array([np.sin(omega*t), 1 - np.cos(omega*t), 0]) + E_cross_B * t)
    return np.array(r_analytical)

################################### Plotting function #################################

def comparaison(L_t, L_r, L_analytic, save = False):
    fig = plt.figure(figsize=(12,5))
    gs = gridspec.GridSpec(2,3,fig,height_ratios=[2,1], hspace=0)
    fig.suptitle("Trajectory and error on each axis")

    ax1, ax2 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[1,0])
    ax3, ax4 = fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,1])
    ax5, ax6 = fig.add_subplot(gs[0,2]), fig.add_subplot(gs[1,2])

    ax1.plot(L_t, L_r[:,0], label="Numerical", color="blue")
    ax1.plot(L_t, L_analytic[:,0], "--",label="Analytic", color="red")
    ax1.set_xticks([])
    ax1.set_ylabel("Displacement")
    ax2.plot(L_t, np.abs((L_r[:,0] - L_analytic[:,0]) / L_analytic[:,0]) * 100, label = "Relativ error", color="green")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Error [%]")

    ax3.plot(L_t, L_r[:,1], label="Numerical", color="blue")
    ax3.plot(L_t, L_analytic[:,1], "--",label="Analytic", color="red")
    ax3.set_xticks([])
    ax4.plot(L_t, np.abs((L_r[:,1] - L_analytic[:,1]) / L_analytic[:,1]) * 100, label = "Relativ error", color="green")
    ax4.set_xlabel("Time [s]")

    ax5.plot(L_t, L_r[:,2], label="Numerical", color="blue")
    ax5.plot(L_t, L_analytic[:,2], "--",label="Analytic", color="red")
    ax5.set_xticks([])
    ax5.legend()
    ax6.plot(L_t, np.abs((L_r[:,2] - L_analytic[:,2]) / L_analytic[:,2]) * 100, label = "Relativ error", color="green")
    ax6.set_xlabel("Time [s]")

    plt.tight_layout()
    if save:
        plt.savefig("comparaison.png", dpi = 300)
    plt.show()

def plot_3component(L_t, L_r, save = False):
    fig = plt.figure(num="3 Component", figsize=(12,5))
    fig.suptitle("All 3 component")
    gs = gridspec.GridSpec(1,3,fig)

    ax1, ax2, ax3 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,2])

    ax1.plot(L_t,L_r[:,0], label="X-component")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Displacement [m]")

    ax2.plot(L_t, L_r[:,1], label = "Y-Component")
    ax2.set_xlabel("Time [s]")
       
    ax3.plot(L_t, L_r[:,2], label = "Z-Component")
    ax3.set_xlabel("Time [s]")

    plt.tight_layout()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    if save:
        plt.savefig("all_component.png", dpi = 300)

def plot_2D(L_x, L_y, save = False):
    plt.figure(num="Plot 2D", figsize=(12,5))
    step = 100
    plt.plot(L_x[step::], L_y[step::], "--",label = "Displacement in time")
    plt.xlabel("X-component [m]")
    plt.ylabel("Y-component [m]")
    plt.title("Plane trajectory")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig("2D_plot.png", dpi = 300)
    plt.show()

def plot_energy(L_t, L_v2, m, save = False):
    L_E = m*m * L_v2 / 2

    plt.figure("Energy", figsize = (12,5))
    plt.plot(L_t, L_E)
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.title(r"Energy conservation ($\vec{E}=0$)")
    plt.tight_layout()

    if save:
        plt.savefig("energy.png",dpi = 300)
    
    plt.show()

#################################### Execution #######################################

q, m = -const.elementary_charge, const.electron_mass

t0, tf, dt = 0, 0.1, 1e-5
r0 = np.array([0.1,0.1,0.1])
v0 = np.array([1e-2,0,0])

L_t, L_r, L_v = Boris_pusher(r0, v0, t0, tf, dt, Electric_field, Magnetic_field, q, m)
L_analytic = analytical_solution(L_t, r0, v0, q, m, Magnetic_field(r0,0), Electric_field(r0,0))
comparaison(L_t, L_r, L_analytic)
#plot_3component(L_t, L_r)
plot_2D(L_r[:,0], L_r[:,1])
plot_energy(L_t, np.linalg.norm(L_v, axis=1), m)