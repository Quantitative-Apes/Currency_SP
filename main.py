import numpy as np
import optimize
import scenarios



optimizer = optimize.TwoStepSPOptimizer()
optimizer.transaction_cost = 0.005
optimizer.gamma = 2

T = 1 # time horizon for simulations
n_scenarios = 10000
r_f = 0.01

scenario_growth_factors, R_f = scenarios.gen_scenarios_normal(scenarios.Consts.test_mu, scenarios.Consts.test_Sigma, n_scenarios, T)


h_init = np.array([0, 0, 0, 0, 0]) # initial holdings
c_init = 1 # initial cash
# TODO: Make Optimizer handle any cash and holding amount, rescaling weights accordingly

res = optimizer.optimize(scenario_growth_factors, R_f, h_init, c_init)
print(res)