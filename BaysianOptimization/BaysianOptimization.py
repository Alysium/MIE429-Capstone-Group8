'''
Implementation of Global Optimization with Gaussian Process

Possible Libraries
- bayes_opt -> import BayesianOptimization

Gaussian Process Notes:
- need to have:
    -> a testable black box function (give inputs and get outputs)
    -> an acquisition funciton (to choose inputs to sample)
    -> an utility function (to optimize and sample with the acquisition funciton)


General Gaussian Processes Steps:
- at each time step, the acquisition function is used to pick a input (or set of inputs) 
  that produces an optimum based on the utility funciton or reveals the most information about the underlying 
  black box function
- as more points gets plotted, a prior is built, resulting in a (hopefully) more accurate postior modelling
  the actual underlying black box function


Current thoughts of Gaussian Processes:
- current difficulty is that there is no blackbox function that can actually be used to get outputs
  from given inputs
- this results in a prior not being able to be built from sampling points via an acquisition funciton
- We are given initial points that we know are on the function (with underlying noise) and can be used for
  a prior; however, these points are not sampled based on the acquisition function
    -> these points are already given with the prior
    -> so: why not just directly model the underlying function with some function (ie: Gaussian, PIecewise, etc)?

- There is no information to go off with in terms of the underlying function to use and would result in a blind guess
- Based on the proposal, it is possible to use proxy functions to proxy performance
    -> 2 concerns with this:
        -> 1): There is no information to go off with in the first place; hence choosing proxy functions is still 
               based on guessing. Sure general function properties can be used (ie: continuous); however, this does
               not cut down the hypothesis space of potential functions to model as proxy functions. Furthermore,
               there is evidence (at least currently) to support more stringent claims of functional properties, 
               which would benefit the proxy functions a bit more
        -> 2): What performance are we evaluating? At this rate, there is currently no performance to evaluate, given
               that the iterations of Baysian Optimization cannot be performed. If we are evaluating the performance
               of fitting functions to the set of data points, then this still raises concerns with (1), given that
               (a) function properties have no evidence and (b) everything can be cherrypicked given there is no
               evidence, resulting in faulty results provided to the client

Thoughts
-> will look to implement Bayesian optimization for a couple function and look at the possibiltiy of convergence
-> possibly see the model/funciton properties from plotting function points?


Functions used
- Rosenbrock
- Schwefel
- Rastrigin

First Goal
-> how to model the data points to generate function(s)

Methods
-> use simple machine learning model (some regression/NN model) to train
  -> possibly introduce more data points with high variance
-> generate some curve function with the data points and use that funciton as the output
-> use funcitonal frameworks and find the error from the data points
  -> ie: linear, quadratic, exponential, etc
  -> if error under a certain threshold (take 25 percentile?) will use in terms of generating output/optimizing
  -> even better: use this error to create weights that are weighted for the function and have their output weighted?

Steps for Implementing Bayesian Optimization
Preliminary)

- How to fill in missing labels?
  -> automatically fill with average of data
    -> can add some noise to the average based on variance
  -> missing value imputation through matrix completion


  -> Imputation with Mean/Median Values
    -> Pros:
      -> works well with small numerical datasets
      -> easy + fast
    -> Cons:
      -> does not factor correlations between features; only works on column level
      -> not very accurate
      -> doesn't account for uncertainty
  -> Fill with zero
  -> Imputation with K-NN
  -> 


1) 
  (i) First figure out how to model the dataPoints or obtain properties of the dataPoints
    -> curve fitting?

    -> Multi Output Models: https://machinelearningmastery.com/multi-output-regression-models-with-python/ 
      -> Idea 1: split 6 dimensional outputs into 6 separate functions where 5 inputs result in an output
        -> justification:
          -> pass through Green Mesa that the outputs are generally independent
          -> although evaluation of the correlation between variables are mixed; some are high, others are low
            -> distance correlation seem to point more towards independence
        -> in this case, still need to determine variable independence
      -> Idea 2: Chained Multioutput
        ->  create linear sequaence of models
        -> given 5 properties, predict 1 property
          -> then use 5 properties  + 1 property to predict 2nd property
          ...
          -> with 6 outputs, there are 6 separate regression problems to model
      

  (ii) Determine the Utility Funciton to be used
2) Either (a) use that model previously as the black box function or (b) use several functions as a supposed black box
  -> this allows us to generate outputs to feed into the utility function to determine what to minimize
3) Iteratively test with different acquisition funcitons and different formulas to calculate expected improvement (if possible)
  -> this allows use to choose between several functions as a "design decsiion"



'''
# Input
# Temperature, Mixer, Amount of Carbon Black, GNP, GNP Production Run Time
# temperature = 0-normal temperature, 1-high temperature
# mixer = 0-normal mixer, 1-different mixer

# Output
# 300% modulus, tensile, elongation, durometer, abrasion resistance, rolling resistance

dataPoints = [
  (1,(0,0,60,0,None),(15.1,20.7,387,72,None,0.282)),
  (2,(0,0,33,3,128),(6,19.9,608,59,None,0.16)),
  (3,(0,0,30,6,128),(5.4,19.7,636,57,None,0.148)),
  (4,(0,0,27,9,128),(5.3,19.2,641,56,None,0.148)),
  (5,(0,0,36,0,None),(6.5,19,566,60,None,0.145)),
  (6,(0,0,40,10,128),(7.6,20.6,602,63,None,0.21)),
  (7,(0,0,35,15,128),(6.9,20.4,650,62,None,0.208)),
  (9,(0,0,45,0,None),(8.8,20.6,514,63,None,0.198)),
  (8,(0,0,45,15,128),(8.8,23.4,637,66,None,0.249)),
  (10,(0,0,45,15,193),(9.2,24.7,625,66,None,0.24)),
  (11,(0,0,45,15,500),(5.9,16,652,67,None,0.318)),
  (12,(1,1,45,15,193),(13.9,22.9,441,65,None,0.178)),
  (13,(0,1,45,15,193),(10.8,22.3,523,67,None,0.222)),
  (14,(0,1,45,15,128),(9.2,22.4,599,66,None,0.245)),
  (15,(0,1,45,15,500),(10,21.7,539,68,None,0.262)),
  (16,(0,1,60,0,None),(13.6,22.8,433,69,None,0.263)),
  (17,(0,1,5,0,None),(6.8,22.6,672,66,None,0.232)),
  (18,(1,1,60,15,193),(14.6,23.2,429,70,None,0.246))
]


class BaysianOptimization:
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