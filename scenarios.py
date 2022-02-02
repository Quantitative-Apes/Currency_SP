import numpy as np
#from numba import jit
class Consts:
    test_mu = np.array([
        0.0437368958795545,
        0.0248231286028781,
        0.0394314007635872,
        0.0478608548410575,
        0.0406759403152394,
    ])

    test_Sigma = np.array(
        [[0.0439707947809288,	0.00789695924566642, 0.0188402562551650, 0.0281448917892919, 0.0208707035295278],
        [0.00789695924566642, 0.0534685292412077, 0.0130152658288257, 0.00561158396343808,	0.00866165463541286],
        [0.0188402562551650,	0.0130152658288257,	0.0823721236886312,	0.0244544002748226,	0.0141911729520033],
        [0.0281448917892919,	0.00561158396343808, 0.0244544002748226, 0.0775116114933941,	0.0306422243431531],
        [0.0208707035295278,	0.00866165463541286, 0.0141911729520033, 0.0306422243431530,	0.0909601659311882]]
    )


def gen_scenarios_normal(params=(Consts.test_mu,Consts.test_Sigma), n=100, T=1, r_f=0.01):
    """Generates scenarios as growth factors from the multivariate normal distribution and a risk free growth factor. Assumes that asset returns follow a log-normal distribution"""
    # TODO: Implement variance reduction techniques
    mu = params[0]
    Sigma = params[1]
    scenario_returns = np.random.multivariate_normal(mean=mu, cov=Sigma, size=n)
    scenario_growth_factors = np.exp(scenario_returns*T)
    R_f = np.exp(r_f*T)
    return scenario_growth_factors, R_f
