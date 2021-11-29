from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import math 
import sys
import matplotlib.pyplot as plt
import random
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import os
from bayes_opt.util import load_logs

dataPoints = [
  (1,(0,0,60,0,270.08960427),(15.1,20.7,387,72,None,0.282)),
  (2,(0,0,33,3,128),(6,19.9,608,59,None,0.16)),
  (3,(0,0,30,6,128),(5.4,19.7,636,57,None,0.148)),
  (4,(0,0,27,9,128),(5.3,19.2,641,56,None,0.148)),
  (5,(0,0,36,0,128),(6.5,19,566,60,None,0.145)),
  (6,(0,0,40,10,128),(7.6,20.6,602,63,None,0.21)),
  (7,(0,0,35,15,128),(6.9,20.4,650,62,None,0.208)),
  (9,(0,0,45,0,215.10743226),(8.8,20.6,514,63,None,0.198)),
  (8,(0,0,45,15,128),(8.8,23.4,637,66,None,0.249)),
  (10,(0,0,45,15,193),(9.2,24.7,625,66,None,0.24)),
  (11,(0,0,45,15,500),(5.9,16,652,67,None,0.318)),
  (12,(1,1,45,15,193),(13.9,22.9,441,65,None,0.178)),
  (13,(0,1,45,15,193),(10.8,22.3,523,67,None,0.222)),
  (14,(0,1,45,15,128),(9.2,22.4,599,66,None,0.245)),
  (15,(0,1,45,15,500),(10,21.7,539,68,None,0.262)),
  (16,(0,1,60,0,176.71396873),(13.6,22.8,433,69,None,0.263)),
  (17,(0,1,5,0,128),(6.8,22.6,672,66,None,0.232)),
  (18,(1,1,60,15,193),(14.6,23.2,429,70,None,0.246))
]

class testFunctions:
  def __init__(self):
    self.dim = 5


  def rosenbrock(self,x1,x2,x3,x4,x5):
    term1 = 100*(x2-x1**2)**2 + (1-x1)**2
    term2 = 100*(x3-x2**2)**2 + (1-x2)**2
    term3 = 100*(x4-x3**2)**2 + (1-x3)**2
    term4 = 100*(x5-x4**2)**2 + (1-x4)**2
    res = sum([term1,term2,term3,term4])
    return -res

  def schwefel(self,x1,x2,x3,x4,x5):
    x = [x1,x2,x3,x4,x5]
    term1 = 418.9829*self.dim
    term2 = sum([xi*math.sin((abs(xi))**0.5) for xi in x])
    res = term1 - term2
    return -res

  def rastrigin(self,x1,x2,x3,x4,x5):
    x = [x1,x2,x3,x4,x5]
    term1 = 10*self.dim
    term2 = sum([xi**2-10*math.cos(2*math.pi*xi) for xi in x])
    res = term1 + term2
    return -res

  def useTestFunction(self, testFunc):
    if testFunc == "rosenbrock":
      return self.rosenbrock
    elif testFunc == "schwefel":
      return self.schwefel
    elif testFunc == "rastrigin":
      return self.rastrigin

  def getInputDomain(self, testFunc):
    if testFunc == "rosenbrock":
      return [(-2.048,2.048) for i in range(self.dim)]
    elif testFunc == "schwefel":
      return [(-500,500) for i in range(self.dim)]
    elif testFunc == "rastrigin":
      return [(-5.12,5.12) for i in range(self.dim)]


class BayesianOptimizationTestFunction:
  def __init__(self, iterations=100, blackBoxFunction = "real", acquisitionFunc="ucb", kappa=2.5, loadData=False):
    self.testFunction = testFunctions()
    self.iterations = int(iterations)
    self.acquisitionFunc = acquisitionFunc
    self.blackBoxFunctionName = blackBoxFunction
    self.kappa=kappa
    if blackBoxFunction == "real":   
      self.blackBoxFunctionUsed = self.blackBox
      self.inputDomain = [(0,1), (0,1), (-100,100),(-100,100),(-100,100)]
    else:
      self.blackBoxFunctionUsed = self.testFunction.useTestFunction(blackBoxFunction)
      self.inputDomain = self.testFunction.getInputDomain(blackBoxFunction)
    self.title = "_".join([self.blackBoxFunctionName.capitalize(), self.acquisitionFunc, str(self.iterations)+"iterations", str(self.kappa)+"kappa"])

    self.loadData = loadData

  def utilityFunction(self, mod300,tensile,elongation,durometer,abrasion,rolling):
    return mod300

  def continuousToDiscrete(self,cts):
    #use step function for cts
    if cts >= 0.5:
      return 1
    return 0

  def blackBox(self,temperature,mixer,cb,gnp,prodTime):
    # Temperature, Mixer, Amount of Carbon Black, GNP, GNP Production Run Time
    temperature = self.continuousToDiscrete(temperature)
    mixer = self.continuousToDiscrete(mixer)

    def f1(temperature,mixer,cb,gnp,prodTime):
      return temperature
    def f2(temperature,mixer,cb,gnp,prodTime,mod300):
      return mod300
    def f3(temperature,mixer,cb,gnp,prodTime,mod300,tensile):
      return mod300
    def f4(temperature,mixer,cb,gnp,prodTime,mod300,tensile,elongation):
      return mod300
    def f5(temperature,mixer,cb,gnp,prodTime,mod300,tensile,elongation,durometer):
      return mod300
    def f6(temperature,mixer,cb,gnp,prodTime,mod300,tensile,elongation,durometer,abrasion):
      return mod300

    mod300 = f1(temperature,mixer,cb,gnp,prodTime)
    tensile = f2(temperature,mixer,cb,gnp,prodTime,mod300)
    elongation = f3(temperature,mixer,cb,gnp,prodTime,mod300,tensile)
    durometer = f4(temperature,mixer,cb,gnp,prodTime,mod300,tensile,elongation)
    abrasion = f5(temperature,mixer,cb,gnp,prodTime,mod300,tensile,elongation,durometer)
    rolling = f6(temperature,mixer,cb,gnp,prodTime,mod300,tensile,elongation,durometer,abrasion)
  
    return self.utilityFunction(mod300,tensile,elongation,durometer,abrasion,rolling)

  def graphTarget(self,iterations,vals):
    plt.plot(iterations, vals)
    plt.xlabel("iterations")
    plt.ylabel("target values")
    plt.title(self.blackBoxFunctionName.capitalize() + " Target Value over "+str(len(iterations)) + " iterations")
    plt.savefig("./graphs/"+self.title+".png")

  def writeData(self, optimalOut, optimalIn):
    with open('./data/BaysianOptimizationResults.txt','a') as f:
      f.write(self.title+": "+ str(optimalOut)+" | "+str(optimalIn))
      f.write("\n")

  # def writeLogFile(self,inputPoints,valsArr):
  #   pass
  #   with open('./manualLogFiles/'+self.title+'.txt', 'a') as f:
  #     for i in range(len(valsArr)):
  #       pass
        

  def run(self, genGraph=True):
    '''
      iterations: int
      blackBoxFunctionUsed: either None or String
        -> None if want to run original black box funciton
        -> String if want to run test black box functions
    
    '''
    print("Running", self.title)
    optimizer = BayesianOptimization(
      f=self.blackBoxFunctionUsed, #can be set to None
      pbounds={
        'x1': self.inputDomain[0], #temperature
        'x2': self.inputDomain[1], #mixer
        'x3': self.inputDomain[2], #cb    #Will need to edit bounds based on requirements
        'x4': self.inputDomain[3], #gnp   #Will need to edit bounds based on requirements
        'x5': self.inputDomain[4] #prodTime #Will need to edit bounds based on requirements
      },
      verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
      random_state=1
    )
    
    #kappa: control balance between exploration and exploitation
    # smaller = prefer exploitation
    # larger = prefer exploration
    random.seed(1)
    print("inital point: ", random.randint(math.floor(self.inputDomain[0][0]), math.ceil(self.inputDomain[0][1])))
    acquisition = UtilityFunction(kind=self.acquisitionFunc, kappa=self.kappa, xi=random.randint(math.floor(self.inputDomain[0][0]), math.ceil(self.inputDomain[0][1])))
    logFlag = False
    try:
      # New optimizer is loaded with previously seen points
      load_logs(optimizer, logs=['./savedModels/'+self.title+'.json'])
      print("loaded")
      logFlag = True
    except Exception as e:
      logger = JSONLogger(path='./savedModels/'+self.title+'.json')
      optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)


    iterArr = []
    inputPoints = []
    valsArr = []
    for i in range(self.iterations):
      next_point = optimizer.suggest(acquisition)
      target = self.blackBoxFunctionUsed(**next_point)
      optimizer.register(params=next_point, target=target)
      iterArr.append(i)
      inputPoints.append(next_point)
      valsArr.append(target)
      if abs(target) < 0.01:
        print ("Stopped Early at iteration", i)
        break
      if i % 250 == 0:
        print("iteration "+str(i)+":", "target:",target,"next point:",next_point)
    
    print("Finished Running")

    if genGraph:
      self.graphTarget(iterArr,valsArr)
    self.writeData(optimizer.max['target'],optimizer.max['params'])
    if logFlag:
      pass
      #self.writeLogFile(inputPoints,valsArr)

if __name__ == "__main__":

  '''
  Run file with "python3 BaysianOptimization <black box function> <number of iterations>
  
  black box functions:
    - "real"
    - "rosenbrock"
    - "schwefel"
    - "rastrigin"
  
  acquisition functions:
    - "ucb": Upper Confidence Bound
    - "ei": Expected Improvement
    - "poi": Probability of Improvement

  '''

  boClass = BayesianOptimizationTestFunction()
  arguements = sys.argv
  kappa = 2.5
  if arguements[1] != "multiple":
    iterations = int(arguements[-1])
    functionToRun = arguements[1]
    acquisitionFunc = arguements[2]
    boClass = BayesianOptimizationTestFunction(iterations=iterations, blackBoxFunction=functionToRun, acquisitionFunc=acquisitionFunc, kappa=kappa)  
    boClass.run()

  else:
    kappaList = [2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5]
    iterationsList = [10000,10000,10000,10000,10000,10000,10000,10000,10000]
    functionsToRunList = ['rosenbrock','rosenbrock','rosenbrock','rastrigin','rastrigin','rastrigin','schwefel','schwefel','schwefel']
    acquisitionFuncsList = ['ucb', 'ei', 'poi', 'ucb', 'ei', 'poi','ucb', 'ei', 'poi',]
    for i in range(len(kappaList)):
      c = BayesianOptimizationTestFunction(iterations=iterationsList[i], blackBoxFunction=functionsToRunList[i], acquisitionFunc=acquisitionFuncsList[i], kappa=kappaList[i])
      c.run()
      print("------")    
