import pickle
import numpy as np
from skopt import gp_minimize
from skopt.plots import plot_convergence
import math
import sys
import time
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
import pickle

#saving progress: https://scikit-optimize.github.io/stable/auto_examples/interruptible-optimization.html


noise_level = 1e-10


# def blackBoxFunciton(x):
#     #x -> 
#     XXXXXXXXX
#     [1 2 3 4 5 6]

#     utility()
    

#     return 3.13




class testFunctions:
    def __init__(self):
        self.dim = 5


    def rosenbrock(self,x,noise_level=noise_level):
        #global minima at (1,1,1,1,1) => f(x) = 0
        term1 = 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
        term2 = 100*(x[2]-x[1]**2)**2 + (1-x[1])**2
        term3 = 100*(x[3]-x[2]**2)**2 + (1-x[2])**2
        term4 = 100*(x[4]-x[3]**2)**2 + (1-x[3])**2
        res = sum([term1,term2,term3,term4])
        return res #scalar

    def schwefel(self,x,noise_level=noise_level):
        term1 = 418.9829*len(x)
        term2 = sum([xi*math.sin((abs(xi))**0.5) for xi in x])
        res = term1 - term2
        return res


    def rastrigin(self,x,noise_level=noise_level):
        term1 = 10*len(x)
        term2 = sum([xi**2-10*math.cos(2*math.pi*xi) for xi in x])
        res = term1 + term2
        return res

    def useTestFunction(self, testFunc):
        if testFunc == "rosenbrock":
            return self.rosenbrock
        elif testFunc == "schwefel":
            return self.schwefel
        elif testFunc == "rastrigin":
            return self.rastrigin
        else:
            print("Incorrect Test Function Passed")
            raise Exception

    def getInputDomain(self, testFunc):
        if testFunc == "rosenbrock":
            return [(-2.048,2.048) for i in range(self.dim)]
        elif testFunc == "schwefel":
            return [(-500,500) for i in range(self.dim)]
        elif testFunc == "rastrigin":
            return [(-5.12,5.12) for i in range(self.dim)]


def boSklearn(iters,acqFunc,random_state=1, funcName="actualFunction",blackBox = None):
    #acq func  need to be: "EI", "LCB", "PI"
    if blackBox != None:
        blackBox = blackBox
    else:
        tFunc = testFunctions()
        blackBox = tFunc.useTestFunction(funcName)
        bounds = tFunc.getInputDomain(funcName)

    runTitle = funcName + "_" + str(iters) + "_" + acqFunc+"_seed"+str(random_state)

    print("------Running Optimization------")
    start_time = time.time()
    checkpoint_saver = CheckpointSaver("./checkpoints/"+runTitle+".pkl", compress=9)
    res = gp_minimize(blackBox,
        bounds,
        acq_func=acqFunc,
        n_calls = iters,
        n_random_starts = 5,
        noise=noise_level,
        callback=[checkpoint_saver],
        random_state=random_state,
        verbose = True,
        kappa = 1 #default = 1.96

    )
    #print("res", res)
    print("Minimum Input: ", res.x)
    print("Minimum Value: ", res.fun)
    print("Finished Running after seconds:", time.time()-start_time)
    print("------Finished Optimization------")
    with open("./datapoints/"+runTitle+".txt","w") as txt_file:
        for i in range(len(res.x_iters)):
            line,yVal = res.x_iters[i], res.func_vals[i]
            txt_file.write(",".join([str(i) for i in line]+[str(yVal)]) + "\n")

    plot_convergence(res)

if __name__ == "__main__":
    

    # print("seen")
    boSklearn(10,"gp_hedge",3, funcName="rastrigin")
    # boSklearn(500,"gp_hedge",3, funcName="rosenbrock")
    # boSklearn(500,"gp_hedge",1, funcName="rastrigin")
    # boSklearn(500,"gp_hedge",1, funcName="schwefel")
    

    #boSklearn("schwefel",500,"gp_hedge",2)
    #boSklearn("schwefel",500,"gp_hedge",3)


    #first run sklear
    #boSklearn("rosenbrock",750,"gp_hedge")
    #boSklearn("schwefel",500,"gp_hedge") #ran to ~550 iters
    #print("rastrigin")
    #boSklearn("rastrigin",750,"gp_hedge")


# if __name__=="__main__":
#     tFunc = testFunctions()
#     args = sys.argv
#     funcName = args[1]
#     iters = int(args[2])
#     acqFunc = args[3] #needs to be "EI", "LCB", "PI"
#     blackBox = tFunc.useTestFunction(funcName)
#     bounds = tFunc.getInputDomain(funcName)

#     runTitle = funcName + "_" + str(iters) + "_" + acqFunc

#     print("------Running Optimization------")
#     start_time = time.time()
#     checkpoint_saver = CheckpointSaver("./checkpoints/"+runTitle+".pkl", compress=9)
#     res = gp_minimize(blackBox,
#         bounds,
#         acq_func=acqFunc,
#         n_calls = iters,
#         n_random_starts = 5,
#         noise=noise_level,
#         callback=[checkpoint_saver],
#         random_state=1,
#         verbose = True,
#         kappa = 1 #default = 1.96
#     )
#     #print("res", res)
#     print("Minimum Input: ", res.x)
#     print("Minimum Value: ", res.fun)
#     print("Finished Running after seconds:", time.time()-start_time)
#     print("------Finished Optimization------")

#     plot_convergence(res)

