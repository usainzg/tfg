from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.model.problem import Problem
import wrm 

from pymoo.algorithms.nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.util.running_metric import RunningMetric

problem = wrm.WRM()

ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=12)
algorithm = NSGA3(pop_size=2000,
                  ref_dirs=ref_dirs)
res = minimize(problem,
               algorithm,
               seed=1,
               termination=('n_gen', 25),
               save_history=True,
               verbose=True)

running = RunningMetric(delta_gen=5,
                        n_plots=3,
                        only_if_n_plots=True,
                        key_press=False,
                        do_show=True)

for algorithm in res.history[:20]:
    running.notify(algorithm)



        