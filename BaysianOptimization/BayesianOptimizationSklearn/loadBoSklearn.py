from skopt import load
import numpy as np
from skopt import gp_minimize
from skopt.plots import plot_convergence
import math
import sys
import time
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
from skopt import gp_minimize
from boSklearn import testFunctions

noise_level = 1e-10
def loadBoSklearn(funcName,iters,acqFunc):
    runTitle = funcName + "_" + str(iters) + "_" + acqFunc
    testFunc = testFunctions()
    blackBox = testFunc.useTestFunction(funcName)
    bounds = testFunc.getInputDomain(funcName)
    


    res = load("./checkpoints/"+runTitle+".pkl")
    print("------Running Optimization------")
    print(res.fun)
    print(len(res.x_iters))
    x0 = res.x_iters
    y0 = res.func_vals
    checkpoint_saver = CheckpointSaver("./checkpoints/"+runTitle+"_reloaded2.pkl", compress=9)
    start_time = time.time()

    newRes = gp_minimize(
        blackBox,
        bounds,
        x0=x0,
        y0=y0,
        acq_func=acqFunc,
        n_calls=iters,
        n_random_starts = 5,
        noise = noise_level,
        callback=[checkpoint_saver],
        random_state = 1,
        verbose=True,
        kappa=4

    )

    print("Minimum Input: ", newRes.x)
    print("Minimum Value: ", newRes.fun)
    print("Finished Running after seconds:", time.time()-start_time)
    print("------Finished Optimization------")

    plot_convergence(newRes)





if __name__ == "__main__":
    args = sys.argv
    funcName = args[1]
    iters = int(args[2])
    acqFunc = args[3] #needs to be "EI", "LCB", "PI", "gp_hedge"

    runTitle = funcName + "_" + str(iters) + "_" + acqFunc


    # res = load("./checkpoints/"+runTitle+".pkl")
    # print(res.fun)
    loadBoSklearn(funcName,iters,acqFunc)


    