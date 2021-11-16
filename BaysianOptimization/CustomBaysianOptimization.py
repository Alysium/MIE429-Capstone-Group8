from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import math 
import sys
import matplotlib.pyplot as plt
class CustomBaysianOptimization:
  def __init__(self):
    pass

  def objectiveFunction(self):
    '''
    input: the 5 rubber property inputs to be used during rubber making
    output: the 6 predicted property outputs to be generated from "tests"

    Thought:
      -> is it possible to split the 6 properties, such that have 6 different functions with 5 inputs -> 1 output?
        -> are properties independent?


    '''
    pass

    

  def utilityFunction(self):
    '''
    input: the 6 outputs from the black box model
    output: a utility score to maximize, ideally between 0 and 1
    '''  
    pass

  def acquisitionFuncition(self):
    '''
    input: 
      -> the property inputs that weere bassed into the obejctive functoin
      -> the resulting output from the utilityFunction
      -> any additional perameters
    
    output:
      -> score of expected improvement/for point at X
    
    '''
    pass

  def proposeLocation(self):
    '''
    input:
    
    output: the next property input to use in objective function
    '''
    pass

  def run(self):
    '''
    the function that runs the overall Baysian Optimization Process
    
    '''
    pass