from firedrake import *
from .classes_phsystem import SysPhdaeRig, \
    check_positive_matrix, check_skew_symmetry


class FreeEB(SysPhdaeRig):

    def __init__(self, n_el, L, rho, A, E, I):
        mesh = IntervalMesh(n_el, L)
        x = SpatialCoordinate(mesh)

        # Finite element defition
        deg = 3
        Vp = FunctionSpace(mesh, "Hermite", deg)
        Vq = FunctionSpace(mesh, "DG", 1)

        V = Vp * Vq
        n_Vp = Vp.dim()
        n_Vq = Vq.dim()
        n_V = V.dim()

        v = TestFunction(V)
        v_p, v_q = split(v)

        e = TrialFunction(V)
        e_p, e_q = split(e)

        al_p = rho * A * e_p
        al_q = 1. / (E * I) * e_q

        dx = Measure('dx')
        ds = Measure('ds')

        m_form = v_p * al_p * dx + v_q * al_q * dx

        petsc_m = assemble(m_form, mat_type='aij').M.handle
        M = np.array(petsc_m.convert("dense").getDenseArray())

        assert check_positive_matrix(M)

        j_gradgrad = v_q * e_p.dx(0).dx(0) * dx
        j_gradgradIP = -v_p.dx(0).dx(0) * e_q * dx

        j_form = j_gradgrad + j_gradgradIP

        petcs_j = assemble(j_form, mat_type='aij').M.handle
        J = np.array(petcs_j.convert("dense").getDenseArray())

        assert check_skew_symmetry(J)

        b0_Fy = v_p * ds(1)
        b0_Mz = v_p.dx(0) * ds(1)
        bL_Fy = v_p * ds(2)
        bL_Mz = v_p.dx(0) * ds(2)

        B0_Fy = assemble(b0_Fy).vector().get_local().reshape((-1, 1))
        B0_Mz = assemble(b0_Mz).vector().get_local().reshape((-1, 1))
        BL_Fy = assemble(bL_Fy).vector().get_local().reshape((-1, 1))
        BL_Mz = assemble(bL_Mz).vector().get_local().reshape((-1, 1))

        B = np.hstack((B0_Fy, B0_Mz, BL_Fy, BL_Mz))

        SysPhdaeRig.__init__(self, n_V, 0, 0, n_Vp, n_Vq, E=M, J=J, B=B)


class ClampedEB(SysPhdaeRig):

    def __init__(self, n_el, L, rho, A, E, I):
        mesh = IntervalMesh(n_el, L)
        x = SpatialCoordinate(mesh)

        # Finite element defition
        deg = 3
        Vp = FunctionSpace(mesh, "DG", 1)
        Vq = FunctionSpace(mesh, "Hermite", deg)

        V = Vp * Vq
        n_Vp = Vp.dim()
        n_Vq = Vq.dim()
        n_V = V.dim()

        v = TestFunction(V)
        v_p, v_q = split(v)

        e = TrialFunction(V)
        e_p, e_q = split(e)

        al_p = rho * A * e_p
        al_q = 1. / (E * I) * e_q

        dx = Measure('dx')
        ds = Measure('ds')

        m_form = v_p * al_p * dx + v_q * al_q * dx

        petsc_m = assemble(m_form, mat_type='aij').M.handle
        M = np.array(petsc_m.convert("dense").getDenseArray())

        assert check_positive_matrix(M)

        j_divDiv = -v_p * e_q.dx(0).dx(0) * dx
        j_divDivIP = v_q.dx(0).dx(0) * e_p * dx

        j_form = j_divDiv + j_divDivIP

        petcs_j = assemble(j_form, mat_type='aij').M.handle
        J = np.array(petcs_j.convert("dense").getDenseArray())

        assert check_skew_symmetry(J)

        b0_wy = v_q.dx(0) * ds(1)
        b0_phiz = - v_q * ds(1)
        bL_wy = - v_q.dx(0) * ds(2)
        bL_phiz = v_q * ds(2)

        B0_wy = assemble(b0_wy).vector().get_local().reshape((-1, 1))
        B0_phiz = assemble(b0_phiz).vector().get_local().reshape((-1, 1))
        BL_wy = assemble(bL_wy).vector().get_local().reshape((-1, 1))
        BL_phiz = assemble(bL_phiz).vector().get_local().reshape((-1, 1))

        B = np.hstack((B0_wy, B0_phiz, BL_wy, BL_phiz))

        SysPhdaeRig.__init__(self, n_V, 0, 0, n_Vp, n_Vq, E=M, J=J, B=B)
