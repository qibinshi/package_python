"""
Stack WaveDecompNet kernel with
2xCNN and 1xFCNN layers and
perform transfer learning with
data augmented on the fly.

@auther: Qibin Shi (qibins@uw.edu)
"""
import h5py
import copy
import torch
import random
import configparser
import pkg_resources
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from numpy.random import default_rng
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .torch_tools import WaveformDataset, try_gpu, CCMSELoss
from .torch_tools import training_loop_branches_augmentation
from .denoiser_util import mkdir, write_progress
from .autoencoder_1D_models_torch import T_model, SeismogramEncoder, SeismogramDecoder, SeisSeparator


def train(configure_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(configure_file)

    transfer = config.getint('training', 'transfer')
    gpu = config.getint('training', 'gpu')
    gpu_ids = config.get('training', 'gpu_ids')
    gpu_ids = [int(x) for x in gpu_ids.split(',')]
    storage_home = config.get('directories', 'storage_home')
    data_dir = storage_home + config.get('directories', 'data_dir')
    model_dir = storage_home + config.get('directories', 'save_model_dir')
    data_file = config.get('data', 'data_file')
    use_demo = config.getint('data', 'use_demo')
    half_length = config.getint('data', 'half_length')
    strmax = config.getint('data', 'stretch_max')
    npts = config.getint('data', 'npts')
    branch_signal = config.get('data', 'branch_signal')
    branch_noise = config.get('data', 'branch_noise')
    train_size = config.getfloat('training', 'train_size')
    test_size = config.getfloat('training', 'test_size')
    rand_seed1 = config.getint('training', 'rand_seed1')
    rand_seed2 = config.getint('training', 'rand_seed2')
    batch_size = config.getint('training', 'batch_size')
    lr = config.getfloat('training', 'learning_rate')
    epochs = config.getint('training', 'epochs')
    minimum_epochs = config.getint('training', 'minimum_epochs')
    patience = config.getint('training', 'patience')
    pre_trained_denote = pkg_resources.resource_filename(__name__, 'pretrained_models/Denote_weights.pth')
    pre_trained_WaveDecompNet = pkg_resources.resource_filename(__name__, 'pretrained_models/WaveDecompNet_weights.pth')
    demo_train_data = pkg_resources.resource_filename(__name__, 'datasets/demo_train_dataset.hdf5')

    if use_demo:
        wave_raw = demo_train_data
    else:
        wave_raw = data_dir + data_file

    print('transfer', transfer)
    print('gpu', gpu)
    print('use demo data', use_demo)
    print('dataset path', wave_raw)
    print('directory to save model', model_dir)
    print('half of the length of waveform', half_length)
    print('fraction for training', train_size)
    print('fraction for testing', test_size * (1 - train_size))
    print('random seeds', rand_seed1, rand_seed2)
    print('batch size', batch_size)
    print('learning rate', lr)
    print('# epochs', epochs)
    print('min. # epochs', minimum_epochs)
    print('patience before stopping', patience)
    print('gpu IDs', gpu_ids)

    mid_pt = half_length

    progress_file = model_dir + '/Running_progress.txt'
    mkdir(model_dir)
    frac = 0.1  # starting fraction not included in shifting window
    weighted_loss = False
    bottleneck_name = 'LSTM'
    model_structure = "Branch_Encoder_Decoder"
    model_name = model_structure + "_" + bottleneck_name

    print("#" * 12 + " Load and split data " + "#" * 12)
    # %% Read the pre-processed dataset and split into train, validate and test sets
    training_data, validate_data, test_data = read_split_data(wave_raw, branch_signal, branch_noise, npts,
                                                              train_size, test_size, rand_seed1, rand_seed2)

    train_iter = DataLoader(training_data, batch_size=batch_size, shuffle=False)
    validate_iter = DataLoader(validate_data, batch_size=batch_size, shuffle=False)
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # %% Fix seed for model initialization
    random.seed(0)
    np.random.seed(20)
    torch.manual_seed(99)
    torch.backends.cudnn.benchmark = False

    ############ %% Neural Net structure %% ###############
    print("#" * 12 + " Loading model " + model_name + " " + "#" * 12)
    devc = try_gpu(i=gpu)

    # %% construct a WaveDecompNet kernel first
    model = WDN_compose()

    if transfer:
        # %% keep constructing for DenoTe
        model = T_model(model, half_insize=int(npts / 2))

        # %% load pre-trained weights for DenoTe
        if torch.cuda.device_count() > gpu:
            model.load_state_dict(torch.load(pre_trained_denote))

            # %% Data parallelism for multiple GPUs,
            # %! model=model.module.to(device) for application on CPU
            if torch.cuda.device_count() > 1 and len(gpu_ids) > 1:
                print("Available number of GPUs", torch.cuda.device_count(), "Let's use GPU:", gpu_ids)
                model = nn.DataParallel(model, device_ids=gpu_ids)
            model.to(devc)
        else:
            model.load_state_dict(torch.load(pre_trained_denote, map_location=devc))

    else:
        # %% load weights for WaveDecompNet
        if torch.cuda.device_count() > gpu:
            model.load_state_dict(torch.load(pre_trained_WaveDecompNet))
            # %% freeze these weights
            # for param in model.parameters():
            #     param.requires_grad = False

            # %% Wrap as DenoTe, with a pre-trained kernel
            model = T_model(model, half_insize=int(npts / 2))

            # %% Data parallelism for multiple GPUs,
            # %! model=model.module.to(device) for application on CPU
            if torch.cuda.device_count() > 1 and len(gpu_ids) > 1:
                print("Available number of GPUs", torch.cuda.device_count(), "Let's use GPU:", gpu_ids)
                model = nn.DataParallel(model, device_ids=gpu_ids)
            model.to(devc)
        else:
            model.load_state_dict(torch.load(pre_trained_WaveDecompNet, map_location=devc))

    n_para = 0
    for idx, param in enumerate(model.parameters()):
        if not param.requires_grad:
            print(idx, param.shape)
        else:
            n_para += np.prod(param.shape)
    print(f'Number of parameters to be trained: {n_para}\n')

    # %% Hyper-parameters for training
    loss_fn = CCMSELoss(use_weight=weighted_loss)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # %% Loop for training
    print("#" * 12 + " training model " + model_name + " " + "#" * 12)

    model, avg_train_losses, avg_valid_losses, partial_loss = training_loop_branches_augmentation(train_iter,
                                                                                                  validate_iter,
                                                                                                  model,
                                                                                                  loss_fn,
                                                                                                  optimizer,
                                                                                                  epochs=epochs,
                                                                                                patience=patience,
                                                                                                  device=devc,
                                                                                          minimum_epochs=minimum_epochs,
                                                                                                    npts=npts,
                                                                                                  mid_pt=mid_pt,
                                                                                                  strmax=strmax)
    print("Training is done!")
    write_progress(progress_file, text_contents="Training is done!" + '\n')

    # %% Save the model
    if torch.cuda.device_count() > gpu and torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), model_dir + f'/{model_name}_weights.pth')
    else:
        torch.save(model.state_dict(), model_dir + f'/{model_name}_weights.pth')

    # %% Save the training history
    loss = avg_train_losses
    val_loss = avg_valid_losses
    with h5py.File(model_dir + f'/{model_name}_Training_history.hdf5', 'w') as f:
        f.create_dataset("loss", data=loss)
        f.create_dataset("val_loss", data=val_loss)
        if model_structure == "Branch_Encoder_Decoder":
            f.create_dataset("earthquake_loss", data=partial_loss[0])
            f.create_dataset("earthquake_val_loss", data=partial_loss[1])
            f.create_dataset("noise_loss", data=partial_loss[2])
            f.create_dataset("noise_val_loss", data=partial_loss[3])

    # %% Save the training info
    with h5py.File(model_dir + f'/{model_name}_Dataset_split.hdf5', 'w') as f:
        f.attrs['model_name'] = model_name
        f.attrs['train_size'] = train_size
        f.attrs['test_size'] = test_size
        f.attrs['rand_seed1'] = rand_seed1
        f.attrs['rand_seed2'] = rand_seed2

    # Calculate the test loss
    test_loss = 0.0
    model.eval()
    for X0, y0 in test_iter:
        nbatch = X0.size(0)
        std_wgt = torch.ones(nbatch, dtype=torch.float64)
        quak2 = torch.zeros(nbatch, y0.size(1), npts * 2, dtype=torch.float64)
        quake = torch.zeros(y0.size(), dtype=torch.float64)
        stack = torch.zeros(y0.size(), dtype=torch.float64)

        rng = default_rng(17)
        rng_snr = default_rng(23)
        rng_sqz = default_rng(11)
        start_pt = rng.choice(npts - int(npts * frac * 2.0), nbatch) + int(npts * frac)
        snr = 10 ** rng_snr.uniform(-0.3, 1, nbatch)
        sqz = rng_sqz.choice(strmax, nbatch) + 1
        pt1 = mid_pt - sqz * npts
        pt2 = mid_pt + sqz * npts

        for i in np.arange(nbatch):
            # %% squeeze earthquake signal
            quak2[i] = X0[i, :, pt1[i]:pt2[i]:sqz[i]]
            # %% shift earthquake signal
            tmp = quak2[i, :, start_pt[i]:start_pt[i] + npts]
            for j in np.arange(X0.size(1)):
                quake[i, j] = torch.div(torch.sub(tmp[j], torch.mean(tmp[j], dim=-1)),
                                        torch.std(tmp[j], dim=-1) + 1e-12) * snr[i]
            # %% stack signal and noise
            stack[i] = quake[i] + y0[i]
            # %% normalize
            scale_mean = torch.mean(stack[i], dim=1)
            scale_std = torch.std(stack[i], dim=1) + 1e-12
            std_wgt[i] = torch.nanmean(scale_std)
            for j in np.arange(X0.size(1)):
                stack[i, j] = torch.div(torch.sub(stack[i, j], scale_mean[j]), scale_std[j])
                quake[i, j] = torch.div(torch.sub(quake[i, j], scale_mean[j]), scale_std[j])

        X, y = stack.to(devc), quake.to(devc)
        snr = torch.from_numpy(snr).to(devc)
        std_wgt = std_wgt.to(devc)

        if len(y.data) != batch_size:
            break
        output1, output2 = model(X)
        loss_pred = loss_fn(output1, y, snr**2) + loss_fn(output2, X - y) + loss_fn(output1 + output2, X, std_wgt**2)
        test_loss += loss_pred.item() * X.size(0)

    test_loss = test_loss/len(test_iter.dataset)

    # %% Show loss evolution when training is done
    plt.close('all')
    plt.figure()
    plt.plot(loss, 'o', label='loss')
    plt.plot(val_loss, '-', label='Validation loss')
    plt.plot([len(loss)], [test_loss], 'r*', label=f'Test loss = {test_loss:.4f}', markersize=10, linewidth=2, zorder=10)

    loss_name_list = ['earthquake train loss', 'earthquake valid loss', 'noise train loss', 'noise valid loss']
    loss_plot_list = ['o', '', 'o', '']
    for ii in range(4):
        plt.plot(partial_loss[ii], marker=loss_plot_list[ii], label=loss_name_list[ii])

    plt.legend()
    plt.title(model_name)
    plt.savefig(model_dir + f'/{model_name}_Training_history.png')


def test(configure_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(configure_file)

    retrain = config.getint('testing', 'retrain')
    retrained_weights = config.get('testing', 'retrained_weights')
    gpu = config.getint('training', 'gpu')
    gpu_ids = config.get('training', 'gpu_ids')
    gpu_ids = [int(x) for x in gpu_ids.split(',')]
    storage_home = config.get('directories', 'storage_home')
    data_dir = storage_home + config.get('directories', 'data_dir')
    model_dir = storage_home + config.get('directories', 'save_model_dir')
    data_file = config.get('data', 'data_file')
    use_demo = config.getint('data', 'use_demo')
    half_length = config.getint('data', 'half_length')
    strmax = config.getint('data', 'stretch_max')
    npts = config.getint('data', 'npts')
    branch_signal = config.get('data', 'branch_signal')
    branch_noise = config.get('data', 'branch_noise')
    train_size = config.getfloat('training', 'train_size')
    test_size = config.getfloat('training', 'test_size')
    rand_seed1 = config.getint('training', 'rand_seed1')
    rand_seed2 = config.getint('training', 'rand_seed2')
    batch_size = config.getint('testing', 'batch_size')
    lr = config.getfloat('training', 'learning_rate')
    epochs = config.getint('training', 'epochs')
    minimum_epochs = config.getint('training', 'minimum_epochs')
    patience = config.getint('training', 'patience')
    pre_trained_denote = pkg_resources.resource_filename(__name__, 'pretrained_models/Denote_weights.pth')
    demo_train_data = pkg_resources.resource_filename(__name__, 'datasets/demo_train_dataset.hdf5')

    if use_demo:
        wave_raw = demo_train_data
    else:
        wave_raw = data_dir + data_file

    if retrain:
        denote_weights = retrained_weights
    else:
        denote_weights = pre_trained_denote

    print('gpu', gpu)
    print('use demo data', use_demo)
    print('dataset path', wave_raw)
    print('directory to load your model', model_dir)
    print('half of the length of waveform', half_length)
    print('fraction for training', train_size)
    print('fraction for testing', test_size * (1 - train_size))
    print('random seeds', rand_seed1, rand_seed2)
    print('batch size', batch_size)
    print('learning rate', lr)
    print('# epochs', epochs)
    print('min. # epochs', minimum_epochs)
    print('patience before stopping', patience)
    print('gpu IDs', gpu_ids)

    mid_pt = half_length

    fig_dir = model_dir + '/figures'
    mkdir(fig_dir)
    dt = 0.1
    frac = 0.45  # smallest window start
    batch_size = 100
    bottleneck_name = 'LSTM'
    model_structure = "Branch_Encoder_Decoder"
    model_name = model_structure + "_" + bottleneck_name

    print("#" * 12 + " Load and split data " + "#" * 12)
    # %% Read the pre-processed dataset and split into train, validate and test sets
    training_data, validate_data, test_data = read_split_data(wave_raw, branch_signal, branch_noise, npts,
                                                              train_size, test_size, rand_seed1, rand_seed2)

    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    ############ %% Neural Net structure %% ###############
    print("#" * 12 + " Loading model " + model_name + " " + "#" * 12)
    devc = torch.device('cpu')

    # %% construct a WaveDecompNet kernel first
    model = WDN_compose()

    # %% keep constructing for DenoTe
    model = T_model(model, half_insize=int(npts / 2))

    # %% load pre-trained weights for DenoTe
    model.load_state_dict(torch.load(denote_weights, map_location=devc))
    model.eval()


    print("#" * 12 + " Augment data and feed to the model " + "#" * 12)
    with torch.no_grad():
        data_iter = iter(test_iter)
        X0, y0 = data_iter.next()

        # %% Augmentation: 1-squeeze 2-shift 3-stack
        nbatch = X0.size(0)
        rng = default_rng(17)
        rng_snr = default_rng(23)
        rng_sqz = default_rng(11)
        start_pt = rng.choice(npts - int(npts * frac * 2), nbatch) + int(npts * frac)
        snr = 10 ** rng_snr.uniform(-0.3, 0.5, nbatch)
        sqz = rng_sqz.choice(strmax, nbatch) + 1
        pt1 = mid_pt - sqz * npts
        pt2 = mid_pt + sqz * npts

        quak2 = torch.zeros(nbatch, y0.size(1), npts * 2, dtype=torch.float64)
        quake = torch.zeros(y0.size(), dtype=torch.float64)
        stack = torch.zeros(y0.size(), dtype=torch.float64)

        for i in np.arange(nbatch):
            # %% squeeze earthquake signal
            print(pt1[i],pt2[i],sqz[i])
            quak2[i] = X0[i, :, pt1[i]:pt2[i]:sqz[i]]
            # %% shift earthquake signal
            tmp = quak2[i, :, start_pt[i]:start_pt[i] + npts]
            for j in np.arange(X0.size(1)):
                quake[i, j] = torch.div(torch.sub(tmp[j], torch.mean(tmp[j], dim=-1)),
                                        torch.std(tmp[j], dim=-1) + 1e-12) * snr[i]
            # %% stack signal and noise
            stack[i] = quake[i] + y0[i]
            # %% normalize
            scale_mean = torch.mean(stack[i], dim=1)
            scale_std = torch.std(stack[i], dim=1) + 1e-12
            for j in np.arange(X0.size(1)):
                stack[i, j] = torch.div(torch.sub(stack[i, j], scale_mean[j]), scale_std[j])
                quake[i, j] = torch.div(torch.sub(quake[i, j], scale_mean[j]), scale_std[j])

        noisy_input, quake_label = stack.to(devc), quake.to(devc)
        noise_label = noisy_input - quake_label

        # %% denoise
        quake_denoised, noise_output = model(noisy_input)
        print("Denoised!")

        noisy_signal = noisy_input.cpu().numpy()
        clean_signal = quake_label.cpu().numpy()
        separated_noise = noise_output.cpu().numpy()
        denoised_signal = quake_denoised.cpu().numpy()
        true_noise = noise_label.cpu().numpy()

    # %% Parallel Plotting
    partial_func = partial(plot_testing, directory=fig_dir, dt=dt, npts=npts)
    num_proc = min(os.cpu_count(), batch_size)
    pool = Pool(processes=num_proc)
    print("#" * 12 + " Plotting waveforms using " + str(num_proc) + "CPUs" + "#" * 12)

    result = pool.starmap(partial_func, zip(noisy_signal,
                                            denoised_signal,
                                            separated_noise,
                                            clean_signal,
                                            true_noise,
                                            np.arange(nbatch),
                                            sqz))

    print("#" * 12 + " Plotting statistics of testing performance " + "#" * 12)
    # %% get the scores for each thread
    # scores = np.zeros((0, 3, 4), dtype=np.double)
    #################################################
    scores = np.zeros((0, 3, 6), dtype=np.double)   #
    #################################################
    for i in range(batch_size):
        scores = np.append(scores, result[i], axis=0)
        print(i, 'th score added')

    # %% Plot statistics of scores
    plt.close("all")
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].hist(scores[:, :, 0].flatten(), bins=10, density=True, histtype='step', color='r', rwidth=0.1, label='EV_quake')
    ax[0].hist(scores[:, :, 1].flatten(), bins=10, density=True, histtype='step', color='k', rwidth=0.1, label='EV_noise')
    ###########################################################
    ax[0].hist(scores[:, :, 4].flatten(), bins=10, density=True, histtype='step', color='g', rwidth=0.1, label='EV_resquake')
    ###########################################################
    ax[0].set_xlabel('explained variance', fontsize=14)
    ax[0].set_ylabel('density', fontsize=14)
    ax[0].legend(loc=2)
    ax[1].hist(scores[:, :, 2].flatten(), bins=10, density=True, histtype='step', color='r', rwidth=0.1, label='CC_quake')
    ax[1].hist(scores[:, :, 3].flatten(), bins=10, density=True, histtype='step', color='k', rwidth=0.1, label='CC_noise')
    ###########################################################
    ax[1].hist(scores[:, :, 5].flatten(), bins=10, density=True, histtype='step', color='g', rwidth=0.1, label='CC_resquake')
    ###########################################################
    ax[1].set_xlabel('cross-correlation coefficient', fontsize=14)
    ax[1].legend(loc=2)

    plt.savefig(fig_dir + f'/statistics_of_scores.pdf')


def plot_testing(noisy_signal, denoised_signal, separated_noise, clean_signal, true_noise, idx, sqz, directory=None, dt=0.1, npts=None):
    quake_denoised = torch.from_numpy(denoised_signal)
    quake_label = torch.from_numpy(clean_signal)
    noise_output = torch.from_numpy(separated_noise)
    noise_label = torch.from_numpy(true_noise)
    #######################################################
    residual_signal = noisy_signal - separated_noise      #
    quake_residual = torch.from_numpy(residual_signal)    #
    #######################################################
    ev_score = Explained_Variance_score()
    loss_fn = CCLoss()
    comps = ['T', 'R', 'Z']
    # scores = np.zeros((1, 3, 4))
    ##################################
    scores = np.zeros((1, 3, 6))     #
    ##################################
    time = np.arange(0, npts) * dt
    gs_kw = dict(height_ratios=[1, 1, 1, 2, 2])

    plt.close("all")
    fig, ax = plt.subplots(5, 3, gridspec_kw=gs_kw, figsize=(12, 12), constrained_layout=True)

    for i in range(3):
        scaling_factor = np.max(abs(noisy_signal[i, :]))
        _, spect_noisy_signal = waveform_fft(noisy_signal[i, :] / scaling_factor, dt)
        _, spect_clean_signal = waveform_fft(clean_signal[i, :] / scaling_factor, dt)
        _, spect_true_noise = waveform_fft(true_noise[i, :] / scaling_factor, dt)
        _, spect_noise = waveform_fft(separated_noise[i, :] / scaling_factor, dt)
        freq, spect_denoised_signal = waveform_fft(denoised_signal[i, :] / scaling_factor, dt)
        ###############################################################################
        _, spect_resquake = waveform_fft(residual_signal[i, :] / scaling_factor, dt)  #
        ###############################################################################
        evs_earthquake = ev_score(quake_denoised[i, :], quake_label[i, :])
        evs_noise = ev_score(noise_output[i, :], noise_label[i, :])
        cc_quake = 1 - loss_fn(quake_denoised[i, :], quake_label[i, :])
        cc_noise = 1 - loss_fn(noise_output[i, :], noise_label[i, :])

        ############################################################################
        evs_resquake = ev_score(quake_residual[i, :], quake_label[i, :])           #
        cc_resquake = 1 - loss_fn(quake_residual[i, :], quake_label[i, :])         #
        ############################################################################

        scores[0, i, 0] = evs_earthquake
        scores[0, i, 1] = evs_noise
        scores[0, i, 2] = cc_quake
        scores[0, i, 3] = cc_noise
        ##################################
        scores[0, i, 4] = evs_resquake   #
        scores[0, i, 5] = cc_resquake    #
        ##################################

        ax[i, 0].plot(time, noisy_signal[i, :] / scaling_factor, '-k', label='Noisy signal', linewidth=1)
        ax[i, 0].plot(time, clean_signal[i, :] / scaling_factor, '-r', label='True signal', linewidth=1)
        ax[i, 1].plot(time, clean_signal[i, :] / scaling_factor, '-r', label='True signal', linewidth=1)
        ax[i, 1].plot(time, denoised_signal[i, :] / scaling_factor, '-b', label='Predicted signal', linewidth=1)
        #################################################################################################
        ax[i, 1].plot(time, residual_signal[i, :] / scaling_factor, '-', color='orange', linewidth=0.3)   #
        #################################################################################################
        ax[i, 2].plot(time, true_noise[i, :] / scaling_factor, '-', color='gray', label='True noise', linewidth=1)
        ax[i, 2].plot(time, separated_noise[i, :] / scaling_factor, '-b', label='Predicted noise', linewidth=1)
        ax[3, i].loglog(freq, spect_noisy_signal, '-k', label='raw signal', linewidth=0.5, alpha=1)
        ax[3, i].loglog(freq, spect_clean_signal, '-r', label='true earthquake', linewidth=0.5, alpha=1)
        ax[3, i].loglog(freq, spect_denoised_signal, '-b', label='separated earthquake', linewidth=0.5, alpha=1)
        ############################################################################################################
        ax[3, i].loglog(freq, spect_resquake, '-', color='orange', label='residual quake', linewidth=0.5, alpha=0.3) #
        ############################################################################################################

        ax[4, i].loglog(freq, spect_true_noise, '-r', label='orginal noise', linewidth=0.5, alpha=1)
        ax[4, i].loglog(freq, spect_noise, '-b', label='noise', linewidth=0.5, alpha=0.8)
        ax[i, 1].text(0, 0.8, f'EV: {evs_earthquake:.2f}/ CC: {cc_quake:.2f}')
        ax[i, 2].text(0, 0.8, f'EV: {evs_noise:.2f}/ CC: {cc_noise:.2f}')

        ax[i, 0].set_ylabel(comps[i], fontsize=16)
        ax[4, i].set_xlabel('Frequency (Hz)', fontsize=14)
        ax[3, i].set_title(comps[i], fontsize=16)
        ax[3, i].grid(alpha=0.2)
        ax[4, i].grid(alpha=0.2)

        for j in range(3):
            ax[i, j].xaxis.set_visible(False)
            ax[i, j].yaxis.set_ticks([])
            ax[i, j].spines['right'].set_visible(False)
            ax[i, j].spines['left'].set_visible(False)
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['bottom'].set_visible(False)

            if i == 2:
                ax[i, j].xaxis.set_visible(True)
                ax[i, j].spines['bottom'].set_visible(True)
                ax[i, j].set_xlabel('time (s)', fontsize=14)
            if i <= 2:
                ax[i, j].set_xlim(0, npts * dt)
                ax[i, j].set_ylim(-1, 1)

    ax[0, 0].set_title("Original signal", fontsize=16)
    ax[0, 1].set_title(f"P wave squeezed x {sqz}")
    ax[0, 2].set_title("Separated noise", fontsize=16)
    ax[3, 0].set_ylabel('velocity spectra', fontsize=14)
    ax[4, 0].set_ylabel('velocity spectra', fontsize=14)
    ax[3, 2].legend(loc=3)
    ax[4, 2].legend(loc=3)

    plt.savefig(directory + f'/time_spec_{idx}.pdf')

    return scores


def read_split_data(wave_raw, branch_signal, branch_noise, npts, train_size, test_size, rand_seed1, rand_seed2):
    # %% Read the pre-processed datasets
    print("#" * 12 + " Loading quake signals and pre-signal noises " + "#" * 12)
    with h5py.File(wave_raw, 'r') as f:
        X_train = f[branch_signal][:]
        Y_train = f[branch_noise][:, (0 - npts):, :]

    X_sum = np.sum(np.sum(np.square(X_train), axis=1), axis=1)
    ind_X = np.where(X_sum == np.nan)[0]
    X_train = np.delete(X_train, ind_X, 0)
    Y_train = np.delete(Y_train, ind_X, 0)
    X_sum = np.sum(np.square(X_train), axis=1)
    ind_X = np.unique(np.where(X_sum == 0)[0])
    X_train = np.delete(X_train, ind_X, 0)
    Y_train = np.delete(Y_train, ind_X, 0)

    print("#" * 12 + " Normalizing signal and noises " + "#" * 12)
    X_train = (X_train - np.mean(X_train, axis=1, keepdims=True)) / (np.std(X_train, axis=1, keepdims=True) + 1e-12)
    Y_train = (Y_train - np.mean(Y_train, axis=1, keepdims=True)) / (np.std(Y_train, axis=1, keepdims=True) + 1e-12)

    # %% Split datasets into train, valid and test sets
    X_training, X_test, Y_training, Y_test = train_test_split(X_train, Y_train,
                                                              train_size=train_size,
                                                              random_state=rand_seed1)
    X_validate, X_test, Y_validate, Y_test = train_test_split(X_test, Y_test,
                                                              test_size=test_size,
                                                              random_state=rand_seed2)

    # %% Convert to torch class. Or WaveformDataset_h5 for limited memory
    training_data = WaveformDataset(X_training, Y_training)
    validate_data = WaveformDataset(X_validate, Y_validate)
    test_data = WaveformDataset(X_test, Y_test)

    return training_data, validate_data, test_data

def WDN_compose():
    # %% construct a simple WaveDecompNet
    bottleneck = torch.nn.LSTM(64, 32, 2, bidirectional=True, batch_first=True, dtype=torch.float64)
    bottleneck_earthquake = copy.deepcopy(bottleneck)
    bottleneck_noise = copy.deepcopy(bottleneck)

    encoder = SeismogramEncoder()
    decoder_earthquake = SeismogramDecoder(bottleneck=bottleneck_earthquake)
    decoder_noise = SeismogramDecoder(bottleneck=bottleneck_noise)

    model = SeisSeparator('WDN', encoder, decoder_earthquake, decoder_noise)

    return model


if __name__ == '__main__':
    train()
    test()
