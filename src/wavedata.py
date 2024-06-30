# https://github.com/sachabinder/wave_equation_simulations/blob/main/2D_WAVE-EQ_variable-velocity.py


"""
This file was built to solve numerically a classical PDE, 2D wave equation. The equation corresponds to

$\dfrac{\partial}{\partial x} \left( \dfrac{\partial c^2 U}{\partial x} \right) + \dfrac{\partial}{\partia
l y} \left( \dfrac{\partial c^2 U}{\partial y} \right) = \dfrac{\partial^2 U}{\partial t^2}$

where
 - U represent the signal
 - x represent the position
 - t represent the time
 - c represent the velocity of the wave (depends on space parameters)

The numerical scheme is based on finite difference method. This program is also providing several boundary conditions. More particularly the Neumann, Dirichlet and Mur boundary conditions.
Copyright - © SACHA BINDER - 2021
"""


############## MODULES IMPORTATION ###############
import numpy as np
import torch


# Def of the initial condition
def I(x, y):
    """
    two space variables depending function
    that represent the wave form at t = 0
    """
    return 0.2 * np.exp(-((x - 2.5) ** 2 / 0.1 + (y - 2.5) ** 2 / 0.1))


def V(x, y):
    """
    initial vertical speed of the wave
    """
    return 0


############## SET-UP THE PROBLEM ###############

# Def of velocity (spatial scalar field)
def celer(x, y):
    """
    constant velocity
    """
    return 1


def single_step(u_n, u_nm1, n, N_x, N_y, source_x, source_y, source_freq, dx, dy, dt, c=1):
    """

    Args:
        u_n:
        u_nm1:
        n: index of timestep
        N_x:
        N_y:
        source_x: x index of source in grid
        source_y: y index of source in grid
        source_freq:
        dx:
        dy:
        dt:
        c: velocity of wave

    Returns:

    """
    Cx2 = (dt / dx) ** 2
    Cy2 = (dt / dy) ** 2

    u_n[source_x, source_y] = np.sin(n * source_freq)

    u_np1 = np.zeros((N_x + 1, N_y + 1), float)

    q = c ** 2 * np.ones((N_x + 1, N_y + 1), float)

    # calculation at step j+1
    # without boundary cond
    u_np1[1:N_x, 1:N_y] = 2 * u_n[1:N_x, 1:N_y] - u_nm1[1:N_x, 1:N_y] + Cx2 * (
            0.5 * (q[1:N_x, 1:N_y] + q[2:N_x + 1, 1:N_y]) * (
            u_n[2:N_x + 1, 1:N_y] - u_n[1:N_x, 1:N_y]) - 0.5 * (q[0:N_x - 1, 1:N_y] + q[1:N_x, 1:N_y]) * (
                    u_n[1:N_x, 1:N_y] - u_n[0:N_x - 1, 1:N_y])) + Cy2 * (
                                  0.5 * (q[1:N_x, 1:N_y] + q[1:N_x, 2:N_y + 1]) * (
                                  u_n[1:N_x, 2:N_y + 1] - u_n[1:N_x, 1:N_y]) - 0.5 * (
                                          q[1:N_x, 0:N_y - 1] + q[1:N_x, 1:N_y]) * (
                                          u_n[1:N_x, 1:N_y] - u_n[1:N_x, 0:N_y - 1]))

    # Nuemann bound cond
    i, j = 0, 0
    u_np1[i, j] = 2 * u_n[i, j] - u_nm1[i, j] + Cx2 * (q[i, j] + q[i + 1, j]) * (
            u_n[i + 1, j] - u_n[i, j]) + Cy2 * (q[i, j] + q[i, j + 1]) * (u_n[i, j + 1] - u_n[i, j])

    i, j = 0, N_y
    u_np1[i, j] = 2 * u_n[i, j] - u_nm1[i, j] + Cx2 * (q[i, j] + q[i + 1, j]) * (
            u_n[i + 1, j] - u_n[i, j]) + Cy2 * (q[i, j] + q[i, j - 1]) * (u_n[i, j - 1] - u_n[i, j])

    i, j = N_x, 0
    u_np1[i, j] = 2 * u_n[i, j] - u_nm1[i, j] + Cx2 * (q[i, j] + q[i - 1, j]) * (
            u_n[i - 1, j] - u_n[i, j]) + Cy2 * (q[i, j] + q[i, j - 1]) * (u_n[i, j - 1] - u_n[i, j])

    i, j = N_x, N_y
    u_np1[i, j] = 2 * u_n[i, j] - u_nm1[i, j] + Cx2 * (q[i, j] + q[i - 1, j]) * (
            u_n[i - 1, j] - u_n[i, j]) + Cy2 * (q[i, j] + q[i, j - 1]) * (u_n[i, j - 1] - u_n[i, j])

    i = 0
    u_np1[i, 1:N_y - 1] = 2 * u_n[i, 1:N_y - 1] - u_nm1[i, 1:N_y - 1] + Cx2 * (
            q[i, 1:N_y - 1] + q[i + 1, 1:N_y - 1]) * (u_n[i + 1, 1:N_y - 1] - u_n[i, 1:N_y - 1]) + Cy2 * (
                                  0.5 * (q[i, 1:N_y - 1] + q[i, 2:N_y]) * (
                                  u_n[i, 2:N_y] - u_n[i, 1:N_y - 1]) - 0.5 * (
                                          q[i, 0:N_y - 2] + q[i, j]) * (
                                          u_n[i, 1:N_y - 1] - u_n[i, 0:N_y - 2]))

    j = 0
    u_np1[1:N_x - 1, j] = 2 * u_n[1:N_x - 1, j] - u_nm1[1:N_x - 1, j] + Cx2 * (
            0.5 * (q[1:N_x - 1, j] + q[2:N_x, j]) * (u_n[2:N_x, j] - u_n[1:N_x - 1, j]) - 0.5 * (
            q[0:N_x - 2, j] + q[1:N_x - 1, j]) * (u_n[1:N_x - 1, j] - u_n[0:N_x - 2, j])) + Cy2 * (
                                  q[1:N_x - 1, j] + q[1:N_x - 1, j + 1]) * (
                                  u_n[1:N_x - 1, j + 1] - u_n[1:N_x - 1, j])

    i = N_x
    u_np1[i, 1:N_y - 1] = 2 * u_n[i, 1:N_y - 1] - u_nm1[i, 1:N_y - 1] + Cx2 * (
            q[i, 1:N_y - 1] + q[i - 1, 1:N_y - 1]) * (u_n[i - 1, 1:N_y - 1] - u_n[i, 1:N_y - 1]) + Cy2 * (
                                  0.5 * (q[i, 1:N_y - 1] + q[i, 2:N_y]) * (
                                  u_n[i, 2:N_y] - u_n[i, 1:N_y - 1]) - 0.5 * (
                                          q[i, 0:N_y - 2] + q[i, 1:N_y - 1]) * (
                                          u_n[i, 1:N_y - 1] - u_n[i, 0:N_y - 2]))

    j = N_y
    u_np1[1:N_x - 1, j] = 2 * u_n[1:N_x - 1, j] - u_nm1[1:N_x - 1, j] + Cx2 * (
            0.5 * (q[1:N_x - 1, j] + q[2:N_x, j]) * (u_n[2:N_x, j] - u_n[1:N_x - 1, j]) - 0.5 * (
            q[0:N_x - 2, j] + q[1:N_x - 1, j]) * (u_n[1:N_x - 1, j] - u_n[0:N_x - 2, j])) + Cy2 * (
                                  q[1:N_x - 1, j] + q[1:N_x - 1, j - 1]) * (
                                  u_n[1:N_x - 1, j - 1] - u_n[1:N_x - 1, j])

    return u_np1


def simulate_wave(L_x, dx, L_y, dy, L_t, dt, source_x, source_y, freq):

    # Spatial mesh - i indices
    # L_x = 5  # Range of the domain according to x [m]
    # dx = 0.05  # Infinitesimal distance in the x direction
    # in code there is +1 added to all Nx and Ny
    # todo change to use N_x directly
    N_x = int(L_x / dx) - 1  # Points number of the spatial mesh in the x direction
    X = np.linspace(0, L_x, N_x + 1)  # Spatial array in the x direction

    # Spatial mesh - j indices
    # L_y = 5  # Range of the domain according to y [m]
    # dy = 0.05  # Infinitesimal distance in the x direction
    N_y = int(L_y / dy) - 1  # Points number of the spatial mesh in the y direction
    Y = np.linspace(0, L_y, N_y + 1)  # Spatial array in the y direction

    assert source_x <= N_x and source_y <= N_y

    # Temporal mesh with CFL < 1 - n indices
    # L_t = 5  # Duration of simulation [s]
    # dt = dt = 0.005#0.1 * min(dx, dy)  # Infinitesimal time with CFL (Courant–Friedrichs–Lewy condition)
    N_t = int(L_t / dt)  # Points number of the temporal mesh
    T = np.linspace(0, L_t, N_t + 1)  # Temporal array

    # Velocity array for calculation (finite elements)
    c = np.ones((N_x + 1, N_y + 1), float)


    ############## CALCULATION CONSTANTS ###############
    Cx2 = (dt / dx) ** 2
    Cy2 = (dt / dy) ** 2
    CFL_1 = dt / dy * c[:, 0]
    CFL_2 = dt / dy * c[:, N_y]
    CFL_3 = dt / dx * c[0, :]
    CFL_4 = dt / dx * c[N_x, :]

    ############## PROCESSING LOOP ###############

    # $\forall i \in {0,...,N_x}$
    U = np.zeros((N_x + 1, N_y + 1, N_t + 1), float)  # Tableau de stockage de la solution

    u_nm1 = np.zeros((N_x + 1, N_y + 1), float)  # Vector array u_{i,j}^{n-1}
    u_n = np.zeros((N_x + 1, N_y + 1), float)  # Vector array u_{i,j}^{n}
    u_np1 = np.zeros((N_x + 1, N_y + 1), float)  # Vector array u_{i,j}^{n+1}
    V_init = np.zeros((N_x + 1, N_y + 1), float)
    q = c ** 2 * np.ones((N_x + 1, N_y + 1), float)

    # init cond - at t = 0
    # for i in range(0, N_x + 1):
    #     for j in range(0, N_y + 1):
    #         q[i, j] = c[i, j] ** 2

    for i in range(0, N_x + 1):
        for j in range(0, N_y + 1):
            V_init[i, j] = V(X[i], Y[j])

    u_n[source_x, source_y] = np.sin(1 * freq)

    U[:, :, 0] = u_n.copy()

    # init cond - at t = 1
    # without boundary cond

    u_np1[1:N_x, 1:N_y] = 2 * u_n[1:N_x, 1:N_y] - (u_n[1:N_x, 1:N_y] - 2 * dt * V_init[1:N_x, 1:N_y]) + Cx2 * (
                0.5 * (q[1:N_x, 1:N_y] + q[2:N_x + 1, 1:N_y]) * (u_n[2:N_x + 1, 1:N_y] - u_n[1:N_x, 1:N_y]) - 0.5 * (
                    q[0:N_x - 1, 1:N_y] + q[1:N_x, 1:N_y]) * (u_n[1:N_x, 1:N_y] - u_n[0:N_x - 1, 1:N_y])) + Cy2 * (
                                      0.5 * (q[1:N_x, 1:N_y] + q[1:N_x, 2:N_y + 1]) * (
                                          u_n[1:N_x, 2:N_y + 1] - u_n[1:N_x, 1:N_y]) - 0.5 * (
                                                  q[1:N_x, 0:N_y - 1] + q[1:N_x, 1:N_y]) * (
                                                  u_n[1:N_x, 1:N_y] - u_n[1:N_x, 0:N_y - 1]))

    # Nuemann bound cond
    i, j = 0, 0
    u_np1[i, j] = 2 * u_n[i, j] - (u_n[i, j] - 2 * dt * V_init[i, j]) + Cx2 * (q[i, j] + q[i + 1, j]) * (
                u_n[i + 1, j] - u_n[i, j]) + Cy2 * (q[i, j] + q[i, j + 1]) * (u_n[i, j + 1] - u_n[i, j])

    i, j = 0, N_y
    u_np1[i, j] = 2 * u_n[i, j] - (u_n[i, j] - 2 * dt * V_init[i, j]) + Cx2 * (q[i, j] + q[i + 1, j]) * (
                u_n[i + 1, j] - u_n[i, j]) + Cy2 * (q[i, j] + q[i, j - 1]) * (u_n[i, j - 1] - u_n[i, j])

    i, j = N_x, 0
    u_np1[i, j] = 2 * u_n[i, j] - (u_n[i, j] - 2 * dt * V_init[i, j]) + Cx2 * (q[i, j] + q[i - 1, j]) * (
                u_n[i - 1, j] - u_n[i, j]) + Cy2 * (q[i, j] + q[i, j + 1]) * (u_n[i, j + 1] - u_n[i, j])

    i, j = N_x, N_y
    u_np1[i, j] = 2 * u_n[i, j] - (u_n[i, j] - 2 * dt * V_init[i, j]) + Cx2 * (q[i, j] + q[i - 1, j]) * (
                u_n[i - 1, j] - u_n[i, j]) + Cy2 * (q[i, j] + q[i, j - 1]) * (u_n[i, j - 1] - u_n[i, j])

    i = 0
    u_np1[i, 1:N_y - 1] = 2 * u_n[i, 1:N_y - 1] - (u_n[i, 1:N_y - 1] - 2 * dt * V_init[i, 1:N_y - 1]) + Cx2 * (
                q[i, 1:N_y - 1] + q[i + 1, 1:N_y - 1]) * (u_n[i + 1, 1:N_y - 1] - u_n[i, 1:N_y - 1]) + Cy2 * (
                                      0.5 * (q[i, 1:N_y - 1] + q[i, 2:N_y]) * (
                                          u_n[i, 2:N_y] - u_n[i, 1:N_y - 1]) - 0.5 * (
                                                  q[i, 0:N_y - 2] + q[i, 1:N_y - 1]) * (
                                                  u_n[i, 1:N_y - 1] - u_n[i, 0:N_y - 2]))

    j = 0
    u_np1[1:N_x - 1, j] = 2 * u_n[1:N_x - 1, j] - (u_n[1:N_x - 1, j] - 2 * dt * V_init[1:N_x - 1, j]) + Cx2 * (
                0.5 * (q[1:N_x - 1, j] + q[2:N_x, j]) * (u_n[2:N_x, j] - u_n[1:N_x - 1, j]) - 0.5 * (
                    q[0:N_x - 2, j] + q[1:N_x - 1, j]) * (u_n[1:N_x - 1, j] - u_n[0:N_x - 2, j])) + Cy2 * (
                                      q[1:N_x - 1, j] + q[1:N_x - 1, j + 1]) * (
                                      u_n[1:N_x - 1, j + 1] - u_n[1:N_x - 1, j])

    i = N_x
    u_np1[i, 1:N_y - 1] = 2 * u_n[i, 1:N_y - 1] - (u_n[i, 1:N_y - 1] - 2 * dt * V_init[i, 1:N_y - 1]) + Cx2 * (
                q[i, 1:N_y - 1] + q[i - 1, 1:N_y - 1]) * (u_n[i - 1, 1:N_y - 1] - u_n[i, 1:N_y - 1]) + Cy2 * (
                                      0.5 * (q[i, 1:N_y - 1] + q[i, 2:N_y]) * (
                                          u_n[i, 2:N_y] - u_n[i, 1:N_y - 1]) - 0.5 * (
                                                  q[i, 0:N_y - 2] + q[i, 1:N_y - 1]) * (
                                                  u_n[i, 1:N_y - 1] - u_n[i, 0:N_y - 2]))

    j = N_y
    u_np1[1:N_x - 1, j] = 2 * u_n[1:N_x - 1, j] - (u_n[1:N_x - 1, j] - 2 * dt * V_init[1:N_x - 1, j]) + Cx2 * (
                0.5 * (q[1:N_x - 1, j] + q[2:N_x, j]) * (u_n[2:N_x, j] - u_n[1:N_x - 1, j]) - 0.5 * (
                    q[0:N_x - 2, j] + q[1:N_x - 1, j]) * (u_n[1:N_x - 1, j] - u_n[0:N_x - 2, j])) + Cy2 * (
                                      q[1:N_x - 1, j] + q[1:N_x - 1, j - 1]) * (
                                      u_n[1:N_x - 1, j - 1] - u_n[1:N_x - 1, j])



    u_nm1 = u_n.copy()
    u_n = u_np1.copy()
    U[:, :, 1] = u_n.copy()

    # Process loop (on time mesh)
    for n in range(2, N_t+1):

        u_np1 = single_step(u_n=u_n, u_nm1=u_nm1, n=n, N_x=N_x, N_y=N_y,
                            source_x=source_x, source_y=source_y, source_freq=freq,
                            dx=dx, dy=dy, dt=dt)

        u_nm1 = u_n.copy()
        u_n = u_np1.copy()
        U[:, :, n] = u_n.copy()

    return U


if __name__ == '__main__':

    num_train_trajectories = 40
    num_test_trajectories = 20

    # fine mesh
    L_x = 5  # Range of the domain according to x [m]
    dx_fine = 0.05  # Infinitesimal distance in the x direction
    dx_coarse = 0.5

    L_y = 5  # Range of the domain according to y [m]
    dy_fine = 0.05  # Infinitesimal distance in the y direction
    dy_coarse = 0.5

    # Temporal mesh with CFL < 1 - n indices
    L_t = 5  # Duration of simulation [s]
    dt = 0.005#0.1 * min(dx, dy)  # Infinitesimal time with CFL (Courant–Friedrichs–Lewy condition)
    subsampling_step = 10  # only take every 10th entry

    freq = 6 * np.pi / 1000  # N_t
    # wavelength: 3 lambda / 5 seconds = c = 1 => lambda = 5/3 = 1.66
    # A = 1  # amplitude of source

    rng_x = np.random.default_rng(seed=10)
    rng_y = np.random.default_rng(seed=11)

    indices_x = np.arange(1, int(L_x / dx_fine))  # indices of all interior mesh points
    indices_y = np.arange(1, int(L_y / dy_fine))

    pos_x_fine = rng_x.choice(indices_x, num_train_trajectories + num_test_trajectories, replace=False)
    pos_y_fine = rng_y.choice(indices_y, num_train_trajectories + num_test_trajectories, replace=False)
    pos_x_coarse = (pos_x_fine * (dx_fine / dx_coarse)).astype(int)
    pos_y_coarse = (pos_y_fine * (dy_fine / dy_coarse)).astype(int)

    for dx, dy, pos_x, pos_y, name in [(dx_fine, dy_fine, pos_x_fine, pos_y_fine, "fine"),
                                       (dx_coarse, dy_coarse, pos_x_coarse, pos_y_coarse, "coarse")]:
        data = []
        for i in range(num_train_trajectories + num_test_trajectories):
            print(i)
            # shape (x_points, y_points, t_points)
            U = simulate_wave(L_x=L_x, dx=dx, L_y=L_y, dy=dy, L_t=L_t, dt=dt,
                              source_x=pos_x[i], source_y=pos_y[i], freq=freq)
            U = np.transpose(U, (2, 0, 1))  # (t_points, x_points, y_points)
            U = U[0::subsampling_step]
            U = np.expand_dims(U, axis=1)  # (t_points, 1, x_points, y_points)
            data.append(U)
        data = torch.tensor(data)
        print(data.shape)  # (num_trajectories, t_points, 1, x_points, y_points)
        data_train = data[:num_train_trajectories]
        data_test = data[num_train_trajectories:]
        torch.save(data_train, f"/home/iris/ppnn/data/{name}_data_train.pt")
        torch.save(data_test, f"/home/iris/ppnn/data/{name}_data_test.pt")

    torch.save(torch.stack((torch.tensor(pos_x_fine[:num_train_trajectories]),
                            torch.tensor(pos_y_fine[:num_train_trajectories])), dim=1),
               "/home/iris/ppnn/data/source_pos_fine_train.pt")
    torch.save(torch.stack((torch.tensor(pos_x_coarse[:num_train_trajectories]),
                            torch.tensor(pos_y_coarse[:num_train_trajectories])), dim=1),
               "/home/iris/ppnn/data/source_pos_coarse_train.pt")
    torch.save(torch.stack((torch.tensor(pos_x_fine[num_train_trajectories:]),
                            torch.tensor(pos_y_fine[num_train_trajectories:])), dim=1),
               "/home/iris/ppnn/data/source_pos_fine_test.pt")
    torch.save(torch.stack((torch.tensor(pos_x_coarse[num_train_trajectories:]),
                            torch.tensor(pos_y_coarse[num_train_trajectories:])), dim=1),
               "/home/iris/ppnn/data/source_pos_coarse_test.pt")
