from firedrake import *
from firedrake.plot import _bezier_calculate_points

import numpy as np
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
from matplotlib import cm

from src.plot_bending import draw_allbending
from src.classes_phsystem import SysPhdaeRig
from src.beam_models import FreeEB, ClampedEB
from src.time_integration import implicit_midpoint

from src.plot_config import configure_matplotlib
configure_matplotlib()
pathout = "figures/"

tev_DAE = np.load("data/t_evDAE.npy")
w0_DAE = np.load("data/w0_solDAE.npy")
ep0_DAE = np.load("data/ep0_solDAE.npy")

n_el = 6
L = 1
frac = 2
per = 1/frac
L1 = per * L
L2 = (1-per) * L

n_el1 = int(n_el/frac)
n_el2 = n_el - n_el1

beamFF = FreeEB(n_el1, L1, 1, 1, 1, 1)
beamCC = ClampedEB(n_el2, L2, 1, 1, 1, 1)

beamFC = SysPhdaeRig.gyrator_ordered(beamFF, beamCC, [2, 3], [0, 1], -np.eye(2))

mesh1 = IntervalMesh(n_el1, L1)
mesh2 = IntervalMesh(n_el2, length_or_left=L1, right=L)

# Finite element defition
deg = 3
Vp1 = FunctionSpace(mesh1, "Hermite", deg)
Vp2 = FunctionSpace(mesh2, "DG", 1)

n_e = beamFC.n
n_p = beamFC.n_p
n_p1 = beamFF.n_p
n_p2 = beamCC.n_p

x1 = SpatialCoordinate(mesh1)[0]
x2 = SpatialCoordinate(mesh2)[0]

omega_r = 4
sqrom_r = np.sqrt(omega_r)

expr_initial_cond = lambda x: 0.5 * (cosh(sqrom_r * x) + cos(sqrom_r * x)) * omega_r
# exp_v1 = 0.5 * (cosh(sqrom_r * x1) + cos(sqrom_r * x1)) * omega_r
# exp_v2 = 0.5 * (cosh(sqrom_r * x2) + cos(sqrom_r * x2)) * omega_r

exp_v1 = expr_initial_cond(x1)
exp_v2 = expr_initial_cond(x2)

v1_0 = Function(Vp1)
v2_0 = Function(Vp2)

v1_0.assign(project(exp_v1, Vp1))
v2_0.assign(interpolate(exp_v2, Vp2))

Mmat = csr_matrix(beamFC.E)
Jmat = csr_matrix(beamFC.J)
Bmat = beamFC.B[:, 2:]

t0 = 0.0
t_fin = 1
n_t = 100
dt = t_fin / n_t

y0 = np.zeros(n_e,)
y0[:n_p1] = v1_0.vector().get_local()
y0[n_p1:n_p] = v2_0.vector().get_local()

input = lambda t: np.array([0.5 * (np.cosh(sqrom_r) + np.cos(sqrom_r)) * omega_r * np.cos(omega_r * t), \
                    sqrom_r/2 * (np.sinh(sqrom_r) - np.sin(sqrom_r)) * omega_r * np.cos(omega_r * t)])

e_sol = implicit_midpoint(y0, Mmat, Jmat, Bmat, input, dt, n_t)

ep_sol = e_sol[:n_p, :]
t_ev = np.linspace(t0, t_fin, num=n_t+1)
dt_vec = np.diff(t_ev)


n_ev = len(t_ev)
w0 = np.zeros((n_p,))
w_sol = np.zeros(ep_sol.shape)
w_sol[:, 0] = w0
w_old = w0


for i in range(1, n_ev):
    w_sol[:, i] = w_old + 0.5 * (ep_sol[:, i - 1] + ep_sol[:, i]) * dt_vec[i-1]
    w_old = w_sol[:, i]


plt.figure()
plt.plot(t_ev, ep_sol[0, :], 'g1', label='Simulated VDD')
plt.plot(tev_DAE, ep0_DAE, 'bo', label='Simulated DAE', mfc='none')
plt.plot(t_ev, omega_r * np.cos(omega_r * t_ev), 'r--', label='Exact')
plt.xlabel('$t \; \mathrm{[s]}$')
plt.ylabel('$e_w(0, t) \; \mathrm{[m/s]}$ ')
plt.title("Vertical velocity at 0")
plt.legend(loc='best')

plt.savefig(pathout + "ref_vel.eps", format="eps")

plt.figure()
plt.plot(t_ev, w_sol[0, :], 'g1', label='Simulated VDD')
plt.plot(tev_DAE, w0_DAE, 'bo', label='Simulated DAE', mfc='none')
plt.plot(t_ev, np.sin(omega_r * t_ev), 'r--', label='Exact')
plt.xlabel('$t \; \mathrm{[s]}$')
plt.ylabel('$w(0, t) \; \mathrm{[m]}$')
plt.title("Vertical displacement at 0")
plt.legend(loc='best')

plt.savefig(pathout + "ref_w.eps", format="eps")


n_plot1 = 15
x1_plot = np.linspace(0, L1, n_plot1)
t_plot = t_ev

v2_f = Function(Vp2)
v2_f.vector().set_local(ep_sol[n_p1:n_p, 0])

v1_i = draw_allbending(n_plot1, [0, 0, 0], ep_sol[:n_p1, 0], L1)[2]

coords_2 = Function(FunctionSpace(mesh2, 'DG', 1))
coords_2.interpolate(x2)
x2_plot_tuple = _bezier_calculate_points(coords_2)
v2_i_tuple = _bezier_calculate_points(v2_f)

x2_plot = np.zeros((2*n_el2))
v2_i = np.zeros((2*n_el2))

for i in range(n_el2):
    x2_plot[2*i] = x2_plot_tuple[i, 1]
    x2_plot[2*i+1] = x2_plot_tuple[i, 0]

    v2_i[2*i] = v2_i_tuple[i, 1]
    v2_i[2*i+1] = v2_i_tuple[i, 0]


x_plot = np.concatenate((x1_plot, x2_plot))

n_plot2 = len(x2_plot)

n_plot = n_plot1 + n_plot2
v_plot = np.zeros((n_plot, n_ev))
w_plot = np.zeros((n_plot, n_ev))

v_plot[:n_plot1, 0] = v1_i
v_plot[n_plot1:n_plot, 0] = v2_i

w1_old = w_plot[:n_plot1, 0]
w2_old = w_plot[n_plot1:n_plot, 0]


for i in range(1, n_ev):
    v1_i = draw_allbending(n_plot1, [0, 0, 0], ep_sol[:n_p1, i], L1)[2]

    v2_f.vector().set_local(ep_sol[n_p1:n_p, i])
    v2_i_tuple = _bezier_calculate_points(v2_f)

    for j in range(n_el2):
        
        v2_i[2*j] = v2_i_tuple[j, 1]
        v2_i[2*j+1] = v2_i_tuple[j, 0]

    v_plot[:n_plot1, i] = v1_i
    v_plot[n_plot1:n_plot, i] = v2_i

    w_plot[:n_plot1, i] = w1_old + 0.5 * (v_plot[:n_plot1, i - 1] + v_plot[:n_plot1, i]) * dt_vec[i-1]
    w_plot[n_plot1:n_plot, i] = w2_old + 0.5 * (v_plot[n_plot1:n_plot, i - 1] + v_plot[n_plot1:n_plot, i]) * dt_vec[i-1]

    w1_old = w_plot[:n_plot1, i]
    w2_old = w_plot[n_plot1:n_plot, i]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X_plot, T_plot = np.meshgrid(x_plot, t_plot)

# Plot the surface.
ax.set_xlabel('$x \; \mathrm{[m]}$', labelpad=10)
ax.set_ylabel('$t \; \mathrm{[s]}$', labelpad=10)
ax.set_zlabel('$w \; \mathrm{[m]}$')

W_plot = np.transpose(w_plot)
surf = ax.plot_surface(X_plot, T_plot, W_plot, cmap=cm.jet,\
                        linewidth=0, antialiased=False, label='Beam $w$')

fig.colorbar(surf, shrink=0.5, aspect=5)
x0 = np.zeros((n_ev,))
w0_plot = ax.plot(x0, t_ev, np.sin(omega_r * t_ev), label='$y = w(0, t)$', color='purple', linewidth=5)

u_t = 0.5 * (np.cosh(sqrom_r) + np.cos(sqrom_r)) * np.sin(omega_r * t_ev)
x1 = np.ones((n_ev,))
w1_plot = ax.plot(x1, t_ev, u_t, label='$u = w(1, t)$', color='black', linewidth=5)

ax.legend(handles=[w0_plot[0], w1_plot[0]], loc="best")

ax.view_init(azim=140)

plt.savefig(pathout + "plotW.eps", format="eps")


plt.show()
