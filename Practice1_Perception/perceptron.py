import numpy as np
import matplotlib.pyplot as plt

# Percerption Test Function
# Group Member: Wei Tianjun / Tang Mingyang


def perceptron_test(seeds=99, num_obsevations=500, learning_rate=1, iteration_thres=0.003):
    print('-----Perception Test-----\nSeed: ' + str(seeds) +
          '\nSample Number: ' + str(num_obsevations*2))
    print('Learning Rate: ' + str(learning_rate) +
          '\nIteration Threshold: ' + str(iteration_thres) + '\n')
    np.random.seed(seeds)
    iteration_error = iteration_thres
    epoch = 0
    # Exit when iteration error smaller than threshold
    while iteration_error >= iteration_thres:
        # Generate random data samples
        iteration_error = .0
        epoch += 1
        vec_x1 = np.random.multivariate_normal(
            [1, 0, -3], [[0, 0, 0], [0, 1, .75], [0, .75, 1]], num_obsevations)
        vec_x2 = np.random.multivariate_normal(
            [1, 1, 2], [[0, 0, 0], [0, 1, .75], [0, .75, 1]], num_obsevations)
        # Initialize parameters
        vec_x = np.vstack((vec_x1, vec_x2)).astype(np.float32)
        vec_y = np.hstack(
            (np.zeros(num_obsevations), np.ones(num_obsevations)))
        vec_w = np.ones(3, dtype=float) / 3
        # Update parameters
        for index in range(vec_x.shape[0]):
            var_y = 1 if np.dot(vec_x[index], vec_w) > 0 else 0
            vec_w = vec_w + learning_rate * \
                (vec_y[index] - var_y) * vec_x[index]
            iteration_error = iteration_error + np.abs(vec_y[index] - var_y)
        iteration_error /= vec_x.shape[0]
        print('Epoch: ' + str(epoch) + ' Error: ' + str(iteration_error))
    # Plot the samples and the linear classifier
    print('-----Iterations Completed-----')
    _ = plt.figure('Perceptron')
    axes = plt.subplot()
    axes.scatter(vec_x[:num_obsevations, 1],
                 vec_x[:num_obsevations, 2], alpha=0.5)
    axes.scatter(vec_x[num_obsevations:, 1],
                 vec_x[num_obsevations:, 2], c='green', alpha=0.5)
    plt.plot([-4, 5],
             [-(1/vec_w[2])*(vec_w[0] + -4 * vec_w[1]), -(1/vec_w[2])*(vec_w[0] + 5 * vec_w[1])])
    plt.annotate(r'$(%.3f)+(%.3f)x+(%.3f)y=0$' %
                 (vec_w[0], vec_w[1], vec_w[2]), xy=(-4, -1.5))
    plt.show()


if __name__ == "__main__":
    perceptron_test()
