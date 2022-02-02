import numpy as np
from tqdm import tqdm

import optimize
import scenarios

def simulate_portfolio(n_scenarios, r_f_opt, r_f_act, T_sim, P, t_p, w_est, t_c_opt, t_c_act, gamma, est_fun, scen_fun, h_init=None, c_init=None):
    """Simulate portfolio over historical prices
    Parameters:
        n_scenarios : amount of scenarios per period
        r_f_opt : yearly risk free interest rate used for parameter estimation
        r_f_act : "actual" yearly risk free interest rate
        T_sim : time horizon for simulations
        P : n x a matrix of prices where n is amount of periods, a is amount of assets
        t_p : time for one period of historical data
        w_est : amount of periods to use for estimation (rolling window used)
        t_c_opt : transaction cost used for optimization
        t_c_act : actual, linear, transaction cost
        gamma : risk aversion parameter
        est_fun : estimation function
        scen_fun : scenario generation function
        h_init (optional) : initial weights in risky assets
        c_init (optional) : initial cash holdings (or stablecoin holding)
    """
    n_periods = P.shape[0]
    n_assets = P.shape[1]
    n_simulation_periods = n_periods - w_est

    if not h_init:
        h_init = np.zeros(n_assets)
    if not c_init:
        c_init = 1

    start_period_idx = w_est
    optimizer = optimize.TwoStepSPOptimizer()
    optimizer.transaction_cost = t_c_opt
    optimizer.gamma = 2

    h = h_init  # NOTE that these are weights and that c+sum(h) should always sum up to one
    c = c_init # NOTE that these are weights and that c+sum(h) should always sum up to one
    assert (sum(h)+c).item() == 1, "Total weights in risk free asset and risky assets should sum up to one"
    wealths = np.zeros(n_simulation_periods) # wealth at start of each period
    holdings = np.zeros((n_simulation_periods, n_assets))
    print('Simulating for', n_simulation_periods, 'periods')

    for i, period in tqdm(enumerate(range(start_period_idx, n_periods)), total=n_simulation_periods):
        if i > 0:
            # calculate returns of our holdings
            c_growth = np.exp(t_p*r_f_act)
            risky_asset_growth = P[period, :]/P[period-1, :]
            portfolio_growth = (risky_asset_growth @ h + c_growth*c).item()
            wealths[i] = portfolio_growth*wealths[i-1]
        else:
            wealths[i] = 1


        # estimate statistics
        P_est = P[period-w_est:period, :]
        dist_params = est_fun(P_est, t_p)
        scenario_growth_factors, R_f = scen_fun(dist_params, n_scenarios, T_sim, r_f_opt)
        x_star, _ = optimizer.optimize(scenario_growth_factors, R_f, h, c)

        # TODO: Oversee this part, not completely accurate
        x_star = x_star*(1-t_c_act) # account for transaction cost
        x_b = x_star[:n_assets]
        x_s = x_star[n_assets:]
        # make the buy/sell decisions
        h = h.reshape(n_assets, 1) - x_s.reshape(n_assets, 1) + x_b.reshape(n_assets, 1) # buy and sell decisions
        holdings[i, :] = h.reshape(n_assets)
        c = c + sum(x_s - x_b) # In reality, we would convert to other currency directly!!!

    
    return wealths, holdings
