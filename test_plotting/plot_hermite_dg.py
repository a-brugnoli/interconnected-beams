import firedrake as fdrk
from firedrake.plot import _bezier_calculate_points
import matplotlib.pyplot as plt
from src.plot_bending import draw_allbending
import numpy as np
from math import pi 

from src.plot_config import configure_matplotlib
configure_matplotlib()

n_elements = 10
L1 = pi
L2 = 2*pi
domain_1 = fdrk.IntervalMesh(n_elements, length_or_left=0, right=L1)
domain_2 = fdrk.IntervalMesh(n_elements, length_or_left=L1, right=L2)

x_coord1 = fdrk.SpatialCoordinate(domain_1)[0]
x_coord2 = fdrk.SpatialCoordinate(domain_2)[0]

hermite_space = fdrk.FunctionSpace(domain_1, "Hermite", 3)
dg1_space = fdrk.FunctionSpace(domain_2, "DG", 1)

v_her = fdrk.Function(hermite_space)
v_dg1 = fdrk.Function(dg1_space)

expression_1 = fdrk.sin(x_coord1)
expression_2 = fdrk.sin(x_coord2)

v_her.assign(fdrk.project(expression_1, hermite_space))
v_dg1.assign(fdrk.interpolate(expression_2, dg1_space))

n_plot1 = 30 

x1_plot = np.linspace(0, L1, n_plot1)
v_her_plot = draw_allbending(n_plot1, [0, 0, 0], v_her.vector().get_local(), L1)[2]

fig, axes = plt.subplots()
plt.plot(x1_plot, v_her_plot, marker="o", color="black", label="hermite plot")
fdrk.plot(v_dg1, axes=axes, color="b", label="fdrk plot")

coords_2 = fdrk.Function(fdrk.FunctionSpace(domain_2, 'DG', 1))
coords_2.interpolate(x_coord2)
x_vals_vdg1 = _bezier_calculate_points(coords_2)
y_vals_vdg1 = _bezier_calculate_points(v_dg1)


plt.plot(x_vals_vdg1, y_vals_vdg1,  marker="+", color = "r", label= "bezier points")
plt.legend()

plt.figure()
for i in range(n_elements):
    plt.plot(x_vals_vdg1[i, :], y_vals_vdg1[i, :], label = f"elem {i+1}")
plt.legend()


# Unpack x vals and y vals

x_vals_vdg1_array = np.zeros((2*n_elements))
y_vals_vdg1_array = np.zeros((2*n_elements))

for i in range(n_elements):
    x_vals_vdg1_array[2*i] = x_vals_vdg1[i, 1]
    x_vals_vdg1_array[2*i+1] = x_vals_vdg1[i, 0]
    y_vals_vdg1_array[2*i] = y_vals_vdg1[i, 1]
    y_vals_vdg1_array[2*i+1] = y_vals_vdg1[i, 0]

plt.figure()

print(x_vals_vdg1_array)
plt.plot(x_vals_vdg1_array, y_vals_vdg1_array, label = "Unpacked points")

plt.show()


