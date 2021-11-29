import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from fancyimpute import KNN
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

dataPoints= [
  (1,(0,0,60,0,0),(15.1,20.7,387,72,-1.63,0.282)),
  (2,(0,0,33,3,128),(6,19.9,608,59,float("NAN"),0.16)),
  (3,(0,0,30,6,128),(5.4,19.7,636,57,float("NAN"),0.148)),
  (4,(0,0,27,9,128),(5.3,19.2,641,56,float("NAN"),0.148)),
  (5,(0,0,36,0,128),(6.5,19,566,60,float("NAN"),0.145)),
  (6,(0,0,40,10,128),(7.6,20.6,602,63,float("NAN"),0.21)),
  (7,(0,0,35,15,128),(6.9,20.4,650,62,float("NAN"),0.208)),
  (9,(0,0,45,0,0),(8.8,20.6,514,63,-1.64,0.198)),
  (8,(0,0,45,15,128),(8.8,23.4,637,66,float("NAN"),0.249)),
  (10,(0,0,45,15,193),(9.2,24.7,625,66,-1.61,0.24)),
  (11,(0,0,45,15,500),(5.9,16,652,67,float("NAN"),0.318)),
  (12,(1,1,45,15,193),(13.9,22.9,441,65,-1.63,0.178)),
  (13,(0,1,45,15,193),(10.8,22.3,523,67,-1.61,0.222)),
  (14,(0,1,45,15,128),(9.2,22.4,599,66,-1.61,0.245)),
  (15,(0,1,45,15,500),(10,21.7,539,68,-1.62,0.262)),
  (16,(0,1,60,0,0),(13.6,22.8,433,69,-1.64,0.263)),
  (17,(0,1,5,0,128),(6.8,22.6,672,66,-1.62,0.232)),
  (18,(1,1,60,15,193),(14.6,23.2,429,70,-1.65,0.246))
]
'''
Possible Paper for discrete/continuous data: http://metalearning.ml/2019/papers/metalearn2019-ru.pdf 
Possible way is to use Binarize categorical/discrete features
    -> use 1/-1 to represent binary features


'''

class dataImputation:
    def __init__(self):
        pass

        
class dataProcessor:
    
    def __init__(self, data, dataType, indices):
        '''
        args:
            dataType: string "input" or "output"
            data: numpy array
            indices: array of indices to consider
        
        '''
        # Input
# Temperature, Mixer, Amount of Carbon Black, GNP, GNP Production Run Time
# temperature = 0-normal temperature, 1-high temperature
# mixer = 0-normal mixer, 1-different mixer

# Output
# 300% modulus, tensile, elongation, durometer, abrasion resistance, rolling resistance
        inputFields = ["Temperature", "Mixer", "Carbon Black", "GNP", "GNP Production Run Time"]
        outputFields = ["300% Modulus", "Tensile","Elongation","Durometer","Abrasion Resistance", "Rolling Resistance"]


        in_out_index = 1
        colNames = inputFields
        if dataType=='output':
            in_out_index = 2
            colNames = outputFields

        data = np.asarray([ [ v for i,v in enumerate(d[in_out_index]) if i in indices] for d in data])
        self.colNames = [v for i,v in enumerate(colNames) if i in indices]
        self.data = data

    def dataImpute(self):
        print(self.data)
        filled = KNN(k=3).fit_transform(self.data)
        print("filled", filled)
    def getDistanceCorrelation(self):
        corr = [[0 for i in range(len(self.colNames))] for i in range(len(self.colNames))]
        for i in range(len(self.colNames)):
            for j in range(len(self.colNames)):
                corr[i][j] = scipy.spatial.distance.correlation(self.data[:,i],self.data[:,j])
        plt.figure()
        sns.heatmap(corr, annot=True)
        plt.savefig("outputDistanceCorr.png")
        return

    def getCorrelation(self, correlationType = 'pearson'):
        '''
        pearson: linear correlation between data samples
        kendall: measure of rank correlation
            -> similarity of orders of the data when ranked by each of the quantities
        spearman: measure of rank correlation
            -> how well the relationship between two variables can be described with monotonic function
        '''
        df = pd.DataFrame(self.data,columns=self.colNames)
        corr=df.corr(method=correlationType).abs()
        plt.subplots(figsize=(13,10))#plt.figure(figsize=None)
        sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns,annot=True)
        plt.savefig("outputCorr.png")

    def getData(self,ind1,ind2):
        return self.data[:,ind1], self.data[:,ind2]

    def plot(self,ind1,ind2):
        d1,d2 = self.getData(ind1,ind2)
        
if __name__ == '__main__':
    outputCorr = dataProcessor(dataPoints, "output", [0,1,2,3,4,5])
    outputCorr.getCorrelation()
    outputCorr.getDistanceCorrelation()
    print("---")
    inputCorr = dataProcessor(dataPoints, 'input', [0,1,2,3,4])
    inputCorr.dataImpute()