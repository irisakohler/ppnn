from types import SimpleNamespace
from gc import collect
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
import yaml

import models
from utility.utils import mesh_convertor, model_count
from src.wavedata import single_step


if __name__ == '__main__':
    # inputfile = sys.argv[1]
    # params = SimpleNamespace(**yaml.load(open(inputfile), Loader=yaml.FullLoader))

    tensorboarddir = "/home/iris/ppnn/wave"
    modelsavepath = "/home/iris/ppnn/wave/model.pth"
    modelsavepath_end_of_training = "/home/iris/ppnn/wave/model_end_of_training.pth"


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
    dt = 0.005#0.1 * min(dx, dy)  # Infinitesimal time with CFL (Courant–Friedrichs–Lewy condition)
    subsampling_step = 10  # only take every 10th entry

    freq = 6 * np.pi / 1000  # N_t

    epochs = 3200
    batchsize = 256
    lr = 1.e-3
    network = "cnn2dwave"

    np.random.seed(12)
    torch.manual_seed(12)
    device = torch.device("cpu")  # todo

    # (num_trajectories, t_points, 1, x_points, y_points)
    coarse_data_train = torch.load("/home/iris/ppnn/data/coarse_data_train.pt", map_location='cpu')
    coarse_data_test = torch.load("/home/iris/ppnn/data/coarse_data_test.pt", map_location='cpu')

    fine_data_train = torch.load("/home/iris/ppnn/data/fine_data_train.pt", map_location='cpu')
    fine_data_test = torch.load("/home/iris/ppnn/data/fine_data_test.pt", map_location='cpu')

    # (num_trajectories, 2)
    source_pos_fine_train = torch.load("/home/iris/ppnn/data/source_pos_fine_train.pt", map_location='cpu')
    source_pos_coarse_train = torch.load("/home/iris/ppnn/data/source_pos_coarse_train.pt", map_location='cpu')
    source_pos_fine_test = torch.load("/home/iris/ppnn/data/source_pos_fine_test.pt", map_location='cpu')
    source_pos_coarse_test = torch.load("/home/iris/ppnn/data/source_pos_coarse_test.pt", map_location='cpu')

    feature_size = [fine_data_train.shape[3], fine_data_train.shape[4]]  # fine mesh size

    timesteps = fine_data_train.shape[1]  # only every 10th timestep was sampled in the data generation

    cmesh = [coarse_data_train.shape[3], coarse_data_train.shape[4]]  # coarsemeshsize
    model_cmesh_size = [25, 25]  # size of coarse mesh in model architecture (not the same as coarse mesh size of solver)
    # modelparams = [cmesh, feature_size]

    # enrich, datachannel = True, 4
    # try:
    #     modelparams += [params.inchannels, ]
    # except AttributeError:
    #     enrich, datachannel = False, 3

    datachannel = 1  # we only have amplitude of wave
    parameterchannel = 2  # x and y index of source positioning

    modelparams = [model_cmesh_size, feature_size, datachannel, parameterchannel]  # assuming parameters have 2 channels

    # parameters
    # pos = torch.linspace(params.para1low, params.para1high, params.num_para1, device=device)
    # Res = torch.linspace(params.para2low, params.para2high, params.num_para2, device=device)
    # pos, Res = torch.meshgrid(pos, Res, indexing='ij')
    # pars = torch.stack((pos, Res), dim=-1).to(device)
    # pars = pars.reshape(-1, 1, 2, 1).repeat(1, timesteps, 1, 1).reshape(-1, 2, 1, 1)

    source_pos_x = source_pos_fine_train[:, 0].to(device)
    source_pos_y = source_pos_fine_train[:, 1].to(device)
    source_pos_x, source_pos_y = torch.meshgrid(source_pos_x, source_pos_y, indexing='ij')
    pars = torch.stack((source_pos_x, source_pos_y), dim=-1).to(device)
    pars = pars.reshape(-1, 1, 2, 1).repeat(1, timesteps, 1, 1).reshape(-1, 2, 1, 1).to(torch.float)
    # convert to float for computation of mean later


    #parstest = torch.tensor([[[[0.5]], [[10000]]]], device=device)
    #begintime = params.datatimestart
    mcvter = mesh_convertor(feature_size, cmesh, dim=2, align_corners=False)
    #coarsesolver = OneStepRunOFCoarse(params.template, params.tmp, params.dt,
    #                                  cmesh, parstest[0, 0].squeeze().item(),
    #                                  1 / parstest[0, 1].squeeze().item(), cmesh[0],
    #                                  params.solver)

    # only take first testing point
    parstest = torch.tensor([[[[source_pos_fine_test[0, 0]]], [[source_pos_fine_test[0, 1]]]]], device=device)

    EPOCH = int(epochs) + 1
    BATCH_SIZE = int(batchsize)

    fdata = fine_data_train.detach().to(torch.float)  # (num_trajectories, t_points, 1, x_points, y_points)

    pdeu_coarse = coarse_data_train.detach().to(torch.float)  # coarse data
    # upsample coarse data to have same size as fine data
    pdeu = mcvter.up(pdeu_coarse.squeeze()).unsqueeze(2)  # todo input tensor in (N, C, d1, d2, ...,dK) format?

    # data_du = (fdata[:, 1:, :3] - pdeu[:, :, :3]).reshape(-1, 3, *feature_size).contiguous()
    # difference between current fine data and coarse data of next step
    #  (num_trajectories * (t_points - 1), 1, x_points, y_points)
    data_du = (fdata[:, 1:] - pdeu[:, :-1]).reshape(-1, 1, *feature_size).contiguous()

    #  (num_trajectories * t_points, 1, x_points, y_points)
    pdeu = pdeu.reshape(-1, datachannel, *feature_size).contiguous()

    #init = fdata[24, :1, :3]
    #label = fdata[24, 1:, :3].detach()
    # test trajectory
    init_0 = fine_data_test[0, 0].to(torch.float)  # initial amplitude T=0
    init_1 = fine_data_test[0, 1].to(torch.float)  # amplitude at T=1
    label = fine_data_test[0, 2:].to(torch.float)  # following trajectory from T=2
    source_pos_coarse_test_single_trajectory = source_pos_coarse_test[0]

    data_u0 = fdata[:, :-1].reshape(-1, datachannel, *feature_size).contiguous()

    fdata = []
    del fdata
    collect()


    def add_plot(p, l=None):  #
        fig, ax = plt.subplots(2, 1, figsize=(5, 5))
        p0 = ax[0].pcolormesh(p, clim=(l.min(), l.max()), cmap='coolwarm')
        fig.colorbar(p0, ax=ax[0])
        if l is not None:
            p2 = ax[1].pcolormesh(l, clim=(l.min(), l.max()), cmap='coolwarm')
            fig.colorbar(p2, ax=ax[1])
        fig.tight_layout()
        return fig


    class myset(torch.utils.data.Dataset):
        def __init__(self):
            self.u0_normd = (data_u0 - data_u0.mean(dim=(0, 2, 3), keepdim=True)) \
                            / data_u0.std(dim=(0, 2, 3), keepdim=True)

            self.outmean = data_du.mean(dim=(0, 2, 3), keepdim=True)
            self.outstd = data_du.std(dim=(0, 2, 3), keepdim=True)
            self.du_normd = (data_du - self.outmean) / self.outstd

            self.parsmean = pars.cpu().mean(dim=(0, 2, 3), keepdim=True)
            self.parsstd = pars.cpu().std(dim=(0, 2, 3), keepdim=True)
            self.pars_normd = (pars.cpu() - self.parsmean) / self.parsstd

            self.pdemean = pdeu.mean(dim=(0, 2, 3), keepdim=True)
            self.pdestd = pdeu.std(dim=(0, 2, 3), keepdim=True)
            self.pdeu_normd = (pdeu - self.pdemean) / self.pdestd

        def __getitem__(self, index):
            return self.u0_normd[index], self.du_normd[index], self.pars_normd[index], self.pdeu_normd[index]

        def __len__(self):
            return self.u0_normd.shape[0]


    dataset = myset()
    inmean, instd = data_u0.mean(dim=(0, 2, 3), keepdim=True).to(device), \
                    data_u0.std(dim=(0, 2, 3), keepdim=True).to(device)
    outmean, outstd = dataset.outmean.to(device), dataset.outstd.to(device)
    parsmean, parsstd = dataset.parsmean.to(device), dataset.parsstd.to(device)

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(tensorboarddir)

    torch.save(inmean, os.path.join(tensorboarddir, 'inmean.pt'))
    torch.save(instd, os.path.join(tensorboarddir, 'instd.pt'))
    torch.save(outmean, os.path.join(tensorboarddir, 'outmean.pt'))
    torch.save(outstd, os.path.join(tensorboarddir, 'outstd.pt'))
    torch.save(parsmean, os.path.join(tensorboarddir, 'parsmean.pt'))
    torch.save(parsstd, os.path.join(tensorboarddir, 'parsstd.pt'))

    pdemean, pdestd = dataset.pdemean.to(device), dataset.pdestd.to(device)
    torch.save(pdemean, os.path.join(tensorboarddir, 'pdemean.pt'))
    torch.save(pdestd, os.path.join(tensorboarddir, 'pdestd.pt'))
    pdeu = []
    del pdeu
    collect()

    data_u0, data_du = [], []
    del data_u0, data_du
    collect()

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True,
                                               num_workers=4)

    model = getattr(models, network)(*modelparams).to(device)

    print('Model parameters: {}\n'.format(model_count(model)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, cooldown=350,
                                                           verbose=True, min_lr=1e-5)
    criterier = nn.MSELoss()

    test_error_best = 1  # 0.5

    for i in range(EPOCH):
        loshis = 0
        counter = 0

        for data in train_loader:

            u0, du, mu, pdeu = data
            u0, du, mu, pdeu = u0.to(device), du.to(device), mu.to(device), pdeu.to(device)

            # if params.noiseinject:
            #     u0 += 0.05 * u0.std() * torch.randn_like(u0)
            #     if params.pde: pdeu += 0.05 * pdeu.std() * torch.randn_like(pdeu)

            u_p = model(u0, mu, pdeu)

            loss = criterier(u_p, du)
            optimizer.zero_grad()
            loss.backward()
            loshis += loss.item()
            optimizer.step()
            counter += 1

        writer.add_scalar('loss', loshis / counter, i)
        scheduler.step(loshis / counter)

        if i % 100 == 0 or i == EPOCH:
            torch.save(model.state_dict(), modelsavepath_end_of_training)

        if i % 100 == 0:
            print('loss: {0:4f}\t epoch:{1:d}'.format(loshis / counter, i))

            model.eval()
            test_re = []
            #u = init[:1].to(device)
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

            for n in range(2, timesteps):

                for substep in range(subsampling_step):
                    # coarse prediction
                    u_np1 = single_step(u_n=u_n.squeeze(), u_nm1=u_nm1.squeeze(), n=n*subsampling_step*dt+substep*dt,
                                        N_x=cmesh[0]-1, N_y=cmesh[1]-1,  # function works with N_x+1
                                        source_x=source_pos_coarse_test_single_trajectory[0],
                                        source_y=source_pos_coarse_test_single_trajectory[1], source_freq=freq,
                                        dx=dx_coarse, dy=dy_coarse, dt=dt)

                    u_nm1 = u_n.copy()
                    u_n = u_np1.copy()  # (x_points, y_points)


                #u_coarse, error = coarsesolver(mcvter.down(u).detach().cpu(), begintime + n * params.dt)
                #if error:
                #    for _ in range(timesteps - n):
                #        test_re.append(float('nan') * torch.ones_like(du))
                #    print('OpenFoam solver failed at step {0}!\n'.format(n))
                #    break
                u_coarse = torch.tensor(u_n).unsqueeze(0).unsqueeze(0).to(torch.float)  # (1, 1, x_points, y_points)
                u_coarse = mcvter.up(u_coarse).to(device)

                du = model(
                    (u - inmean) / instd,
                    (parstest - parsmean) / parsstd,
                    (u_coarse - pdemean) / pdestd,
                ) * outstd + outmean

                u = du + u_coarse#[:, :3]  # todo why 3 ?

                test_re.append(u.detach())

            model.train()

            test_re = torch.cat(test_re, dim=0).cpu()

            for testtime in [0, (timesteps - 1) // 4, (timesteps - 1) // 2, 3 * (timesteps - 1) // 4, -1]:
                writer.add_figure('test result {}'.format(testtime),
                                  add_plot(
                                      (test_re[testtime, 0]),
                                      (label[testtime, 0])),
                                  i)

            test_error = criterier(test_re[-1], label[-1]) / criterier(label[-1], torch.zeros_like(label[-1]))
            #test_error_u = criterier(test_re[-1, 0], label[-1, 0]) / criterier(label[-1, 0],
            #                                                                   torch.zeros_like(label[-1, 0]))
            #test_error_v = criterier(test_re[-1, 1], label[-1, 1]) / criterier(label[-1, 1],
            #                                                                   torch.zeros_like(label[-1, 1]))

            #writer.add_scalar('U rel_error', test_error_u, i)
            #writer.add_scalar('V rel_error', test_error_v, i)
            writer.add_scalar('rel_error', test_error, i)
            if test_error < test_error_best:
                test_error_best = test_error
                torch.save(model.state_dict(), modelsavepath)
    writer.close()
