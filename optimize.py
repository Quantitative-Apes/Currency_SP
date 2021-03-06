import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
#from numba import jit

class TwoStepSPOptimizer:
    """Class to solve two step stochastic programming portfolio investment decision problem"""
    def __init__(self):
        self.transaction_cost = 0.05 # linear transaction costs
        self.gamma = 2 # risk aversion parameter (higher equals more risk averse)
        self.scale = 1


    def optimize(self, scenarios, R_f, h_init, c_init):
        """Determine optimal buy/sell decisions given initial portfolio and generated scenarios. Returns optimal buy/sell decisions"""
        n_assets = scenarios.shape[1]
        h_init = h_init.reshape(n_assets, 1) # should always sum up to one
        x = np.zeros((n_assets*2, 1)) # initial solution

        assert (sum(h_init)+c_init).item() == 1, "Total weights in risk free asset and risky assets should sum up to one"

        # Set up constraints
        shorting_constraint = np.concatenate([-np.eye(n_assets), np.eye(n_assets)], axis=1)
        borrowing_constraint = np.concatenate([np.ones((1, n_assets)), -np.ones((1, n_assets))], axis=1) + self.transaction_cost
        A = np.concatenate([shorting_constraint, borrowing_constraint])

        shorting_bound = h_init
        borrowing_bound = c_init
        b = np.array([np.append(shorting_bound, borrowing_bound)]).squeeze()

        constraints = LinearConstraint(A, -np.ones(len(b))*np.inf, b)#, keep_feasible=True)
        bounds = Bounds(np.zeros(len(x)), np.ones(len(x))*np.inf)#, keep_feasible=True)

        obj = lambda x : self.objective(x, scenarios, R_f, h_init, c_init)
        res = minimize(obj, x, bounds=bounds, constraints=constraints)
        return res.x, res

    def objective(self, x, scenarios, R_f, h_init, c_init):
        """Power utility function objective to minimize"""
        n_scenarios = scenarios.shape[0]
        probabilities = np.ones((n_scenarios, 1))/n_scenarios # Uniform probabilities
        n_assets = int(len(x)/2)
        x_b = x[:n_assets].reshape(n_assets, 1) # buy
        x_s = x[n_assets:].reshape(n_assets, 1) # sell

        h = np.matmul(scenarios, (h_init.reshape(n_assets, 1)+x_b-x_s))
        c = R_f*(c_init + np.sum(x_s-x_b) - self.transaction_cost)
        wealth_scenarios = h + c
        
        probabilities = np.ones((n_scenarios, 1))/n_scenarios # equally probable scenarios
        expected_utility = np.matmul(probabilities.transpose(), (wealth_scenarios**(1-self.gamma))) /(1-self.gamma)
        z = -expected_utility.squeeze() # scalar, minimization
        return z.squeeze()

    def get_tradeable_decisions(x_star, C_init, h_init, t_c_variable = 0.001):
        """Under the assumption that it is cheaper to trade directly between coins instead of selling to base 
        and then buying again, determine optimal decisions. Also minimizes amount of trades"""
        n_assets = len(h_init)
        buy_with_cash = np.zeros(n_assets) # e.g. stablecoin
        buy_with_currency = np.zeros((n_assets, n_assets))



