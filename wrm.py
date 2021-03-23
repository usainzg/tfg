import autograd.numpy as anp
import numpy as np
from pymoo.model.problem import Problem


class WRM(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=5, n_constr=7, type_var=anp.double)
        self.xl = anp.array([0.01, 0.01, 0.01])
        self.xu = anp.array([0.45, 0.10, 0.10])
    
    def _evaluate(self, x, out, *args, **kwargs):
        g1 = (0.00139 /(x[:, 0] * x[:, 1])) + 4.94 * x[:, 2] - 0.08
        g2 = (0.000306 /(x[:, 0] * x[:, 1])) + 1.082 * x[:, 2] - 0.0986
        g3 = (12.307 /(x[:, 0] * x[:, 1])) + 49408.24 * x[:, 2] + 4051.02
        g4 = (2.098 /(x[:, 0] * x[:, 1])) + 8046.33 * x[:, 2] - 696.71
        g5 = (2.138 /(x[:, 0] * x[:, 1])) + 7883.39 * x[:, 2] - 705.04
        g6 = (0.417 /(x[:, 0] * x[:, 1])) + 1721.26 * x[:, 2] - 136.54
        g7 = (0.164 /(x[:, 0] * x[:, 1])) + 631.13 * x[:, 2] - 54.48

        f1 = 106780.37 * (x[:, 1] + x[:, 2]) + 61704.67
        f2 = 3000 * x[:, 0]
        f3 = (305700 * 2289 * x[:, 1]) / ((0.06 * 2289) ** 0.65)
        f4 = 250 * 2289 * np.exp(-39.75 * x[:, 1] + 9.9 * x[:, 2] + 2.74)
        f5 = 25 * (1.39 / (x[:, 0] * x[:, 1]) + 4940 * x[:, 2] - 80)

        g1 = - 1 + g1 / 1.0
        g2 = -1 + g2 / 1.0
        g3 = -1 + g3 / 5000.0
        g4 = -1 + g4 / 16000.0
        g5 = -1 + g5 / 10000.0
        g6 = -1 + g6 / 2000.0
        g7 = -1 + g7 / 550.0

        out["F"] = anp.column_stack([f1, f2, f3, f4, f5])
        out["G"] = anp.column_stack([g1, g2, g3, g4, g5, g6, g7])

