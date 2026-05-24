import numpy as np

#Futuramente colocar aqui as preparacoes para atenuacao, por enquanto sua estrutura é provisoria para apenas compilar o codigo com essa nova estrutura

class AttenuationCoefficients:

    def __init__(self):

        self._n_sls = 2

        self._kappa_coeffs = (
            np.array([1.4338786388150824e-07,1.4940867411163837e-08], dtype=np.float32),# * 0.48 ,
            np.array([1.3978352607844830e-07,1.4559615104162200e-08], dtype=np.float32) #* 0.48
        )

        self._non_kappa_coeffs = (
            np.array([1.6114410440373681E-007,3.1979515203307799E-008], dtype=np.float32),# * 0.48,
            np.array([1.5963189129026462E-007, 3.1793743078764331E-008], dtype=np.float32)# * 0.48
        )

        self._tau_epsilon_p, self._tau_sigma_p = self._kappa_coeffs
        self._tau_epsilon_s, self._tau_sigma_s = self._non_kappa_coeffs

        alpha_p = self._tau_epsilon_p / self._tau_sigma_p
        alpha_s = self._tau_epsilon_s / self._tau_sigma_s
        self._sum_alpha_p = sum(alpha_p)
        self._sum_alpha_s = sum(alpha_s)