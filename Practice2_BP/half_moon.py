import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
import nn_block


# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    var_h = 0.01
    # Generate a grid of points with distance h between them
    var_xx, var_yy = np.meshgrid(np.arange(x_min, x_max, var_h),
                                 np.arange(y_min, y_max, var_h))
    # Predict the function value for the whole gid
    vec_z = pred_func(np.c_[var_xx.ravel(), var_yy.ravel()])
    vec_z = vec_z.reshape(var_xx.shape)
    # Plot the contour and training examples
    plt.contourf(var_xx, var_yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


if __name__ == "__main__":
    # Display plots inline and change default figure size
    matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

    np.random.seed(3)
    X, y = sklearn.datasets.make_moons(200, noise=0.15)

    # Train the logistic rgeression classifier
    CLF = nn_block.NNClassifier(
        input_features=2, hidden_features=3, output_features=2)
    print(X.shape, y.shape)
    CLF.fit(X, y, num_passes=500, batch_size=5, epsilon=0.01)

    # Plot the decision boundary
    plot_decision_boundary(lambda x: CLF.predict(x))
    plt.title("Plot decision boundary with hidden layer size 3")
    plt.show()
