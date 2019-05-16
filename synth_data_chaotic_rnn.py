import numpy as np
import matplotlib.pyplot as plt
import utils

def rateScale(r, maxRate):
    r = 0.5 * maxRate * (r + 1)
    return r
    
def eulerStep(x_prev, xgrad, dt):
    x_next = x_prev + dt * xgrad
    return x_next

def RNNgrad(y, W, tau):
    ygrad = -y + W.dot(np.tanh(y))
    return ygrad / tau
    
def spikify_rates(r, dt):
    # r in Hz, dt in seconds
    s = np.random.poisson(r*dt)
    return s

def split_data(data, split_ix):
    data_train = data[:, :split_ix, :, :]
    data_valid = data[:, split_ix:, :, :]
    return data_train, data_valid

def generate_data(T= 1, dt_rnn= 0.01, dt_cal= 0.1,
                  Ninits= 400, Ntrial= 10, Ncells= 50, trainp= 0.8,
                  tau=0.025, gamma=1.5, maxRate=5, B=20,
                  tau_c = 0.4, inc_c=1.0, sigma=0.2,
                  seed=5, save=False):
    
    '''
    Generate synthetic calcium fluorescence data from chaotic recurrent neural network system
    
    Arguments:
        - T (int or float): total time in seconds to run 
        - dt_rnn (float): time step of chaotic RNN
        - dt_cal (float): time step of calcium trace
        - Ninits (int): Number of network initialisations
        - Ntrial (int): Number of instances with same network initialisations
        - Ncells (int): Number of cells in network
        - trainp (float): proportion of dataset to partition into training set
        - tau (float): time constant of chaotic RNN
        - gamma (float): 
        - maxRate (float): maximum firing rate of chaotic RNN
        - B (int, or float): amplitude of perturbation to network
        - save (bool): save output
    '''
    
    np.random.seed(seed)

    Nsteps = int(T / dt_rnn)
    Ntrial_train = int(trainp * Ntrial)
    
    # Chaotic RNN weight matrix
    W = gamma*np.random.randn(Ncells, Ncells)/np.sqrt(Ncells)

    rates, spikes = np.zeros((2, Ninits, Ntrial, Nsteps, Ncells))
    
    perturb_steps = []

    for init in range(Ninits):
        y0 = np.random.randn(Ncells)

        for trial in range(Ntrial):
            perturb_step = np.random.randint(0.25*Nsteps,0.75*Nsteps)
            perturb_steps.append(perturb_step)
            perturb_amp = np.random.randn(Ncells)*B
            b = 0

            yt = y0
            rt = rateScale(np.tanh(yt), maxRate=maxRate)
            st = spikify_rates(rt, dt=dt_cal)

            rates[init, trial, 0, :]   = rt
            spikes[init, trial, 0, :]  = st

            for step in range(1, Nsteps):
                yt = eulerStep(yt, RNNgrad(yt+b, W, tau), dt_rnn)

                if step == perturb_step:
                    b = perturb_amp*dt_rnn/tau
                else:
                    b = 0

                rt = rateScale(np.tanh(yt), maxRate=maxRate)
                st = spikify_rates(rt, dt=dt_cal)

                rates[init, trial, step, :]   = rt
                spikes[init, trial, step, :]  = st
    
    # Construct data dictionary
    
    rates_train, rates_valid     = split_data(rates, Ntrial_train)
    rates_train = np.reshape(rates_train, (Ninits * Ntrial_train, Nsteps, Ncells))
    rates_valid = np.reshape(rates_valid, (Ninits * (Ntrial-Ntrial_train), Nsteps, Ncells))
    del rates

    spikes_train, spikes_valid   = split_data(spikes, Ntrial_train)
    spikes_train = np.reshape(spikes_train, (Ninits * Ntrial_train, Nsteps, Ncells))
    spikes_valid = np.reshape(spikes_valid, (Ninits * (Ntrial-Ntrial_train), Nsteps, Ncells))
    del spikes

    data_dict = {
        'train_spikes'  : spikes_train,
        'valid_spikes'  : spikes_valid,
        'train_rates'   : rates_train,
        'valid_rates'   : rates_valid,
        'perturb_times' : np.array(perturb_steps)*dt_cal
    }    
    if save:
        utils.write_data('./synth_data/chaotic_rnn_%03d'%seed, data_dict)
        
    return data_dict

