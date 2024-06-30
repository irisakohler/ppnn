from types import SimpleNamespace
from gc import collect
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import yaml

from src import models
from src.utility.utils import mesh_convertor
from src.wavedata import single_step

if __name__ == '__main__':

    # generate long time model prediction for first training trajectory

    tensorboarddir = "/home/iris/ppnn/wave"
    modelsavepath = "/home/iris/ppnn/wave/model.pth"
    modelsavepath_end_of_training = "/home/iris/ppnn/wave/model_end_of_training.pth"
    normpath = "/home/iris/ppn/wave/"
    result_save_path = "/home/iris/ppnn/wave/first_training_point_prediction.pt"

    timestep_start = 0

    # copied from wavedata
    # fine mesh
    L_x = 5  # Range of the domain according to x [m]
    dx_fine = 0.05  # Infinitesimal distance in the x direction
    dx_coarse = 0.5

    L_y = 5  # Range of the domain according to y [m]
    dy_fine = 0.05  # Infinitesimal distance in the y direction
    dy_coarse = 0.5

    # Temporal mesh with CFL < 1 - n indices
    L_t = 5  # Duration of simulation [s]
    dt = 0.005  # 0.1 * min(dx, dy)  # Infinitesimal time with CFL (Courant–Friedrichs–Lewy condition)
    subsampling_step = 10  # only take every 10th entry

    freq = 6 * np.pi / 1000  # N_t

    network = "cnn2dwave"

    np.random.seed(12)
    torch.manual_seed(12)
    device = torch.device("cpu")  # todo

    # (num_trajectories, t_points, 1, x_points, y_points)
    coarse_data_train = torch.load("/home/iris/ppnn/data/coarse_data_train.pt", map_location='cpu')[:, timestep_start:]
    coarse_data_test = torch.load("/home/iris/ppnn/data/coarse_data_test.pt", map_location='cpu')[:, timestep_start:]

    fine_data_train = torch.load("/home/iris/ppnn/data/fine_data_train.pt", map_location='cpu')[:, timestep_start:]
    fine_data_test = torch.load("/home/iris/ppnn/data/fine_data_test.pt", map_location='cpu')[:, timestep_start:]

    # (num_trajectories, 2)
    source_pos_fine_train = torch.load("/home/iris/ppnn/data/source_pos_fine_train.pt", map_location='cpu')
    source_pos_coarse_train = torch.load("/home/iris/ppnn/data/source_pos_coarse_train.pt", map_location='cpu')
    source_pos_fine_test = torch.load("/home/iris/ppnn/data/source_pos_fine_test.pt", map_location='cpu')
    source_pos_coarse_test = torch.load("/home/iris/ppnn/data/source_pos_coarse_test.pt", map_location='cpu')

    feature_size = [fine_data_train.shape[3], fine_data_train.shape[4]]  # fine mesh size

    timesteps = fine_data_train.shape[1]  # only every 10th timestep was sampled in the data generation

    cmesh = [coarse_data_train.shape[3], coarse_data_train.shape[4]]  # coarsemeshsize
    model_cmesh_size = [25,
                        25]  # size of coarse mesh in model architecture (not the same as coarse mesh size of solver)

    datachannel = 1  # we only have amplitude of wave
    parameterchannel = 2  # x and y index of source positioning

    modelparams = [model_cmesh_size, feature_size, datachannel, parameterchannel]  # assuming parameters have 2 channels

    mcvter = mesh_convertor(feature_size, cmesh, dim=2, align_corners=False)

    # use first training point to test model performance
    parstest = torch.tensor([[[[source_pos_fine_train[0, 0]]], [[source_pos_fine_train[0, 1]]]]], device=device)
    init_0 = fine_data_train[0, 0].to(torch.float)  # initial amplitude T=0
    init_1 = fine_data_train[0, 1].to(torch.float)  # amplitude at T=1
    label = fine_data_train[0, 2:].to(torch.float)  # following trajectory from T=2
    source_pos_coarse_test_single_trajectory = source_pos_coarse_train[0]

    inmean = torch.load(os.path.join(normpath, 'inmean.pt'), map_location=device)
    instd = torch.load(os.path.join(normpath, 'instd.pt'), map_location=device)
    outmean = torch.load(os.path.join(normpath, 'outmean.pt'), map_location=device)
    outstd = torch.load(os.path.join(normpath, 'outstd.pt'), map_location=device)
    parsmean = torch.load(os.path.join(normpath, 'parsmean.pt'), map_location=device)
    parsstd = torch.load(os.path.join(normpath, 'parsstd.pt'), map_location=device)
    pdemean = torch.load(os.path.join(normpath, 'pdemean.pt'), map_location=device)
    pdestd = torch.load(os.path.join(normpath, 'pdestd.pt'), map_location=device)

    model = getattr(models, network)(*modelparams).to(device)
    model.load_state_dict(torch.load(modelsavepath_end_of_training, map_location=device))

    model.eval()
    test_re = []
    # shape (1, x_points, y_points)
    # solver input
    u_nm1 = init_0
    u_n = init_1
    # unsqueeze do get right shape for mcvter,
    # downsample and squeeze to remove single dimension for prediction with solver
    u_nm1 = mcvter.down(u_nm1.unsqueeze(0)).squeeze().numpy()
    u_n = mcvter.down(u_n.unsqueeze(0)).squeeze().numpy()

    # model input
    u = init_1.to(device).unsqueeze(0)  # (1, 1, x_points, y_points)

    for n in range(2, 5*timesteps):  # go 5 times beyond training range

        for substep in range(subsampling_step):
            # coarse prediction
            u_np1 = single_step(u_n=u_n.squeeze(), u_nm1=u_nm1.squeeze(), n=n * subsampling_step * dt + substep * dt,
                                N_x=cmesh[0] - 1, N_y=cmesh[1] - 1,  # function works with N_x+1
                                source_x=source_pos_coarse_test_single_trajectory[0],
                                source_y=source_pos_coarse_test_single_trajectory[1], source_freq=freq,
                                dx=dx_coarse, dy=dy_coarse, dt=dt)

            u_nm1 = u_n.copy()
            u_n = u_np1.copy()  # (x_points, y_points)

        u_coarse = torch.tensor(u_n).unsqueeze(0).unsqueeze(0).to(torch.float)  # (1, 1, x_points, y_points)
        u_coarse = mcvter.up(u_coarse).to(device)

        du = model(
            (u - inmean) / instd,
            (parstest - parsmean) / parsstd,
            (u_coarse - pdemean) / pdestd,
        ) * outstd + outmean

        u = du + u_coarse

        test_re.append(u.detach())

    test_re = torch.cat(test_re, dim=0).cpu()
    torch.save(test_re, result_save_path)
