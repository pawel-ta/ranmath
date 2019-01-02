
from Ranmath import TimeSeriesMatrix
from Ranmath.MatrixReconstructors import SingleMatrixReconstructor
from Ranmath.CorrelationEstimators import LinearShrinkageEstimator, LedoitPecheRIEstimator, QuarticRIEstimator, OracleEstimator
from Ranmath.Norms import frobenius_eigenvalues_distance
import builtins
import matplotlib.pyplot as plt, numpy as np

print = lambda x: builtins.print(x, end='\n\n')

print("Creation")

matrix = TimeSeriesMatrix()

#print("Importing")

#matrix.from_CSV("data.csv")

#plot = plt.plot(matrix.array[:5, :].T), plt.xlabel("Loaded data [first 5 plots]")

#plt.show(plot)

print("WISHART")
matrix.generate.inverse_wishart(5, 100, 3.5)

#macierz kowariancji, pierwsze okno, sample
print(matrix.characteristics.rw_covariance_cubes(1, 0).sample_cube[0])

#print(matrix.array)

autocorr_eigvec = matrix.characteristics.rw_autocorrelation_eigenvectors(1, 0)

autocorr_eigval = matrix.characteristics.rw_autocorrelation_eigenvalues(1, 0)

print("RECONSTRUCTING")

#to powinno dac przy generatorach rzeczywista macierz bedaca macierza kowariancji z pierwszego okna, (sample)

reconstructor = SingleMatrixReconstructor(autocorr_eigvec.sample_eigenvectors[0],
                                          autocorr_eigval.sample_eigenvalues[0])

print(reconstructor.reconstruct())

plot = plt.plot(matrix.array[:5, :].T), plt.xlabel("Inverse wishart for 5 assets")

plt.show(plot)

print("Generating")

matrix.generate.exponential_decay(50, 100, 3.5)
print(matrix.array)

plot = plt.plot(matrix.array[:5, :].T), plt.xlabel("Exponential Decay [first 5 plots]")
plt.show(plot)

matrix.generate.inverse_wishart(50, 100, 0.3)
print(matrix.array)

plot = plt.plot(matrix.array[:1, :].T), plt.xlabel("Inverse Wishart [first 5 plots]")
plt.show(plot)

print("Exporting")

matrix.to_CSV("exported.csv")
array = matrix.to_ndarray()
print(array)

print("Getting characteristics")

covariance = matrix.characteristics.rw_covariance_cubes(3, 4)
eigenValues = matrix.characteristics.rw_autocorrelation_eigenvalues(3, 4)
eigenVectors = matrix.characteristics.rw_autocorrelation_eigenvectors(3, 4)

print("Normalizing")

#matrix.normalize.winsorization(False, 0.9)
matrix.normalize.standard()

plot = plt.plot(matrix.array[:1, :].T), plt.xlabel("Inverse Wishart [first 5 plots] normalized")
plt.show(plot)

matrix.normalize.outlier()

ls_estimator = LinearShrinkageEstimator()
lp_estimator = LedoitPecheRIEstimator()
qt_estimator = QuarticRIEstimator()
or_estimator = OracleEstimator()

q=1/2
matrix.generate.inverse_wishart(50, 100, 0.3)
sample_eigvals = matrix.characteristics.rw_autocorrelation_eigenvalues(20, 20).sample_eigenvalues[0]
oos_eigvals = matrix.characteristics.rw_autocorrelation_eigenvalues(20, 20).out_of_sample_eigenvalues[0]

eigvals_est = []

optimal_alpha = ls_estimator.get_bonafide_alpha(matrix.characteristics.rw_data_cubes(20, 20).sample_cube, matrix.characteristics.rw_autocorrelation_eigenvalues(20, 20).sample_eigenvalues)
#optimal_alpha = ls_estimator.get_oracle_alpha(matrix.generate.last_C, matrix.characteristics.rw_covariance_cubes(100,100).sample_cube)


print(optimal_alpha)



oracle_eigvals = or_estimator.estimate_eigenvalues(matrix.characteristics.rw_autocorrelation_eigenvectors(20, 20).sample_eigenvectors[0], matrix.generate.last_C)

alpha_factor_list = np.arange( -0 , 1.05 , 0.05 )
eta_factor_list = 10 ** np.arange( -2. , 1.05 , 0.05 )

#for alpha in alpha_factor_list:
#    eigvals_est.append(ls_estimator.estimate_eigenvalues(sample_eigvals, alpha))

for eta in eta_factor_list:
   eigvals_est.append(qt_estimator.estimate_eigenvalues(sample_eigvals, qt_estimator.get_optimal_q(50, 100), 0.3, eta))

to_plot = np.array([frobenius_eigenvalues_distance(estimated, oracle_eigvals) for estimated in eigvals_est])
fig , axs = plt.subplots( 1 , 2 , sharex = False , sharey = False )

axs[ 0  ].plot( eta_factor_list , to_plot , c = 'purple' , label = 'LSE' )
axs[ 0  ].set_title( "Inverse Wishart κ = 0.3" )
axs[ 0  ].set_xlabel( 'η' )
axs[ 0 ].set_ylabel( 'Eigenvalues Frobenius distance from oracle')

matrix.generate.exponential_decay(50, 100, 3.5)
sample_eigvals = matrix.characteristics.rw_autocorrelation_eigenvalues(20, 20).sample_eigenvalues[0]
oos_eigvals = matrix.characteristics.rw_autocorrelation_eigenvalues(20, 20).out_of_sample_eigenvalues[0]

eigvals_est = []

oracle_eigvals = or_estimator.estimate_eigenvalues(matrix.characteristics.rw_autocorrelation_eigenvectors(20, 20).sample_eigenvectors[0], matrix.generate.last_C)

alpha_factor_list = np.arange( -0 , 1.05 , 0.05 )
eta_factor_list = 10 ** np.arange( -2. , 1.05 , 0.05 )

for eta in eta_factor_list:
    eigvals_est.append(qt_estimator.estimate_eigenvalues(sample_eigvals, qt_estimator.get_optimal_q(50, 100), 0.3, eta))

to_plot = np.array([frobenius_eigenvalues_distance(estimated, oracle_eigvals) for estimated in eigvals_est])

axs[ 1  ].plot( eta_factor_list , to_plot , c = 'purple' , label = 'LSE' )
axs[ 1  ].set_title( "Exponential Decay τ = 3.5" )
axs[ 1  ].set_xlabel( 'η' )
axs[ 1 ].set_ylabel( 'Eigenvalues Frobenius distance from oracle')

plt.show(plot)
