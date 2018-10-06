
from Ranmath import TimeSeriesMatrix

import builtins

print = lambda x: builtins.print(x, end='\n\n')

print("Creation")

matrix = TimeSeriesMatrix()

print("Importing")

matrix.fromCSV("data.csv")
matrix.fromNDArray([])

print("Generating")

matrix.generate.ED(...) #Exponential Decay
matrix.generate.IW(...) #Inverse Wishart
matrix.generate.MVGaussian(...) #Multivariate Gaussian

print("Exporting")

matrix.toCSV("exported.csv")
array = matrix.toNDArray()

print("Normalizing")

matrix.normalize.outlier(...)
matrix.normalize.standard(...)
matrix.normalize.winsorization(...)

print("Getting characteristics")

covariance = matrix.characteristics.covarianceCube(...)
eigenValues = matrix.characteristics.eigenValues(...)
eigenVectors = matrix.characteristics.eigenVectors(...)

