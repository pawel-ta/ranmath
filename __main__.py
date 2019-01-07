
from Ranmath import TimeSeriesMatrix
from Ranmath.MatrixReconstructors import SingleMatrixReconstructor
from Ranmath.CorrelationEstimators import LinearShrinkageEstimator, LedoitPecheRIEstimator, QuarticRIEstimator, OracleEstimator
from Ranmath.Norms import frobenius_eigenvalues_distance
import builtins
import matplotlib.pyplot as plt, numpy as np


def test_generating_inverse_wishart(iterations):
    matrix = TimeSeriesMatrix()
    matrix.generate.inverse_wishart(150, 200, 0.3, iterations, normalise_covariance=True, verbose=True)
    print(matrix.array.shape)
    for iteration in range(iterations):
        plot = plt.plot(matrix.array[iteration, :5, :].T)
        plt.xlabel("Inverse Wishart iteration "+str(iteration)+" [first 5 assets]")
        plt.show(plot)


def test_generating_exp_decay(iterations):
    matrix = TimeSeriesMatrix()
    matrix.generate.exponential_decay(150, 200, 4.75, iterations, verbose=True)
    print(matrix.array.shape)
    for iteration in range(iterations):
        plot = plt.plot(matrix.array[iteration, :5, :].T)
        plt.xlabel("Inverse Wishart iteration "+str(iteration)+" [first 5 assets]")
        plt.show(plot)


def test_saving_and_reading_from_csv():
    matrix = TimeSeriesMatrix()
    matrix.generate.exponential_decay(150, 200, 4.75, 30, verbose=True)
    generated = matrix.array
    matrix.export_matrix.to_CSV("exported.csv")
    matrix.import_matrix.from_CSV("exported.csv")
    imported = matrix.array
    if np.array_equiv(generated[:0, :, :], imported):
        print("PASSED, array exported and loaded correctly")
    else:
        fig, axs = plt.subplots(1, 2, sharex=False, sharey=False)
        axs[0].plot(generated[0, :3, :].T)
        axs[1].plot(imported[0, :3, :].T)
        plt.show()


def test_rolling_window():
    matrix = TimeSeriesMatrix()
    matrix.generate.multivariate_gaussian(np.eye(150), np.eye(200), 10, verbose=True)
    print(matrix.characteristics.rw_autocorrelation_eigenvalues(10, 10, verbose=True).sample_eigenvalues.shape)
    print(matrix.characteristics.rw_autocorrelation_eigenvectors(50, 50, verbose=True).sample_eigenvectors.shape)
    print(matrix.characteristics.rw_sample_estimator_cubes(30, 30, verbose=True).sample_cube.shape)
    print(matrix.characteristics.rw_data_cubes(40, 40, verbose=True).sample_cube.shape)


def test_global_sampler():
    matrix = TimeSeriesMatrix()
    matrix.generate.multivariate_gaussian(np.eye(150), np.eye(200), 10, verbose=True)
    print(matrix.characteristics.global_eigenvalues(verbose=True).shape)
    print(matrix.characteristics.global_eigenvectors(verbose=True).shape)
    print(matrix.characteristics.global_sample_estimator_cube(verbose=True).shape)

if __name__ == '__main__':
    print("Started")
    #test_generating_exp_decay(5) <- works
    #test_saving_and_reading_from_csv() <- works
    #test_saving_and_reading_from_csv() <- works
    #test_rolling_window() <- works
    #test_global_sampler() <- works



