
from Ranmath import TimeSeriesMatrix

import builtins

print = lambda x: builtins.print(x, end='\n\n')

print("Creation")

matrix = TimeSeriesMatrix()

print("Importing")

matrix.fromCSV("data.csv")
print(matrix.array)

print("Generating")

matrix.generate.exponentialDecay(50, 100, 3.5)
print(matrix.array)

matrix.generate.inverseWishart(50, 100, 0.3)
print(matrix.array)


print("Exporting")

matrix.toCSV("exported.csv")
array = matrix.toNDArray()
print(array)

print("Normalizing")

matrix.normalize.outlier(...)
matrix.normalize.standard(...)
matrix.normalize.winsorization(...)

print("Getting characteristics")

covariance = matrix.characteristics.covarianceCube(...)
eigenValues = matrix.characteristics.eigenValues(...)
eigenVectors = matrix.characteristics.eigenVectors(...)

