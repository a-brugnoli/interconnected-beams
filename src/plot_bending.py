# EB beam written with the port Hamiltonian approach

import numpy as np
import scipy.linalg as la



def draw_bending(n_draw, v_rig, v_fl, L):
    # Suppose no displacement in zero
    assert len(v_rig) == 3
    assert len(v_fl) % 2 == 0
    n_el = int(len(v_fl) / 2)

    wfl_dofs = v_fl

    dx_el = L / n_el

    u_P = v_rig[0]
    w_P = v_rig[1]
    th_P = v_rig[2]

    x_coord = np.linspace(0, L, num=n_draw)

    u_r = u_P * np.ones_like(x_coord)
    w_r = w_P * np.ones_like(x_coord) + x_coord * th_P

    w_fl = np.zeros_like(x_coord)

    i_el = 0
    xin_elem = i_el * dx_el

    for i in range(n_draw):

        x_til = (x_coord[i] - xin_elem) / dx_el

        if x_til > 1:
            i_el = i_el + 1
            if i_el == n_el:
                i_el = i_el - 1

            xin_elem = i_el * dx_el
            x_til = (x_coord[i] - xin_elem) / dx_el

        phi1_w = 1 - 3 * x_til ** 2 + 2 * x_til ** 3
        phi2_w = x_til - 2 * x_til ** 2 + x_til ** 3
        phi3_w = + 3 * x_til ** 2 - 2 * x_til ** 3
        phi4_w = + x_til ** 3 - x_til ** 2

        if i_el == 0:
            w_fl[i] = phi3_w * wfl_dofs[2 * i_el] + phi4_w * wfl_dofs[2 * i_el + 1]
        else:
            w_fl[i] = phi1_w * wfl_dofs[2 * (i_el - 1)] + phi2_w * wfl_dofs[2 * i_el - 1] + \
                        phi3_w * wfl_dofs[2 * i_el] + phi4_w * wfl_dofs[2 * i_el + 1]

    u_tot = u_r
    w_tot = w_r + w_fl

    return x_coord, u_tot, w_tot


def draw_allbending(n_draw, v_rig, v_fl, L):
    # Suppose displacement in every point
    assert len(v_rig) == 3
    assert len(v_fl) % 2 == 0
    n_el = int(len(v_fl) / 2) - 1

    wfl_dofs = v_fl

    dx_el = L / n_el

    u_P = v_rig[0]
    w_P = v_rig[1]
    th_P = v_rig[2]

    x_coord = np.linspace(0, L, num=n_draw)

    u_r = u_P * np.ones_like(x_coord)
    w_r = w_P * np.ones_like(x_coord) + x_coord * th_P

    w_fl = np.zeros_like(x_coord)

    i_el = 0
    xin_elem = i_el * dx_el

    for i in range(n_draw):

        x_til = (x_coord[i] - xin_elem) / dx_el

        if x_til > 1:
            i_el = i_el + 1
            if i_el == n_el:
                i_el = i_el - 1

            xin_elem = i_el * dx_el
            x_til = (x_coord[i] - xin_elem) / dx_el

        phi1_w = 1 - 3 * x_til ** 2 + 2 * x_til ** 3
        phi2_w = x_til - 2 * x_til ** 2 + x_til ** 3
        phi3_w = + 3 * x_til ** 2 - 2 * x_til ** 3
        phi4_w = + x_til ** 3 - x_til ** 2

        w_fl[i] = phi1_w * wfl_dofs[2 * i_el] + phi2_w * wfl_dofs[2 * i_el + 1] + \
                    phi3_w * wfl_dofs[2 * i_el + 2] + phi4_w * wfl_dofs[2 * i_el + 3]

    u_tot = u_r
    w_tot = w_r + w_fl

    return x_coord, u_tot, w_tot

