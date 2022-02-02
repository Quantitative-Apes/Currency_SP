import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

import optimize
import scenarios
import simulation
import estimate_statistics

def demo():
    optimizer = optimize.TwoStepSPOptimizer()
    optimizer.transaction_cost = 0.005 # not the actual transaction cost
    optimizer.gamma = 2

    T = 1 # time horizon for simulations
    n_scenarios = 10000
    r_f = 0.01

    scenario_growth_factors, R_f = scenarios.gen_scenarios_normal(scenarios.Consts.test_mu, scenarios.Consts.test_Sigma, n_scenarios, T)


    h_init = np.array([100, 40, 30, 20, 10]) # initial holdings
    c_init = 5000 # initial cash
    #h_init = np.array([0, 0, 0, 0, 0]) # initial holdings
    #c_init = 1 # initial cash





    x_star, res = optimizer.optimize(scenario_growth_factors, R_f, h_init, c_init)
    print(res)


    print(sum(h_init+x_star[0:5]-x_star[5:]))

def simulate():

    P = pd.DataFrame()
    coin_returns = {}
    for coin in ['ADA', 'BNB', 'DASH', 'DOGE', 'ETH']:
        fname = coin+'BUSD_5min.csv'

        df = pd.read_csv(os.path.join(os.getcwd(), 'data', fname))
        P = pd.concat([P, df['close']], axis=1)
    
    P = P.values[-3000:, :]
    coin_returns = P[-1,:]/P[0,:]
    W, H = simulation.simulate_portfolio(
        n_scenarios=10000,
        r_f_opt = 0,
        r_f_act = 0,
        T_sim = 12*5, # expressed in 5minute intervals TODO: Figure out good way to standardize minutely returns to daily
        P = P,
        t_p = 1, # 5 minute period
        w_est = 288, # 1 day period used for statistics estimate
        t_c_opt = 0.0005,
        t_c_act = 0.001, # corresponding to Binance's VIP Tier 0 (base)
        gamma = 2,
        est_fun=estimate_statistics.est_log_normal,
        scen_fun=scenarios.gen_scenarios_normal,
    )

    # TODO: Add plot labels
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(W)
    ax[0].scatter(np.ones(len(coin_returns))*len(W), coin_returns)
    # add corresponding returns for currencies used
    ax[1].plot(H)
    plt.show()

if __name__ == '__main__':

    simulate()