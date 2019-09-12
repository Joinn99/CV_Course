# Package imports
import numpy as np
from sklearn.datasets import fetch_openml


class NNClassifier:
    # Initialize nn model
    def __init__(self, input_features=5, hidden_features=3, output_features=2):
        self.model = {"W1": np.random.random_sample((input_features, hidden_features)),
                      "b1": np.random.random_sample((1, hidden_features)),
                      "W2": np.random.random_sample((hidden_features, hidden_features)),
                      "b2": np.random.random_sample((1, output_features)),
                      "W3": np.random.random_sample((hidden_features, hidden_features)),
                      "b3": np.random.random_sample((1, hidden_features)),
                      "W4": np.random.random_sample((hidden_features, output_features)),
                      "b4": np.random.random_sample((1, output_features)),
                      }

    # Helper function to predict an output (0 or 1)
    def predict(self, x):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        W3, b3, W4, b4 = self.model['W3'], self.model['b3'], self.model['W4'], self.model['b4']
        # Forward propagation
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        a2 = np.tanh(z2)
        z3 = a2.dot(W3) + b3
        a3 = np.tanh(z3)
        z4 = a3.dot(W4) + b4
        exp_scores = np.exp(z4)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(np.array(probs), axis=1)

    def _calculate_loss(self, X, y, reg_lambda):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        W3, b3, W4, b4 = self.model['W3'], self.model['b3'], self.model['W4'], self.model['b4']
        # Forward propagation to calculate our predictions
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        a2 = np.tanh(z2)
        z3 = a2.dot(W3) + b3
        a3 = np.tanh(z3)
        z4 = a3.dot(W4) + b4
        exp_scores = np.exp(z4)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        corect_logprobs = 0 - np.log(probs[range(y.shape[0]), y])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss
        data_loss += reg_lambda / 2 * \
            (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1. / y.shape[0] * data_loss

    def fit(self, X, y, num_passes=2000, epsilon=0.01, reg_lambda=0.01, batch_size=500):
        epoch = 0
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        W3, b3, W4, b4 = self.model['W3'], self.model['b3'], self.model['W4'], self.model['b4']
        # Gradient descent. For each batch...
        print('Start training.')
        for _ in range(0, num_passes):
            epoch += 1
            batch_start = 0
            while batch_start < X.shape[0]:
                if batch_start + batch_size >= X.shape[0]:
                    batch_X = X[batch_start:, :]
                    batch_y = y[batch_start:]
                else:
                    batch_X = X[batch_start:batch_start+batch_size, :]
                    batch_y = y[batch_start:batch_start+batch_size]
                # Forward propagation
                z1 = batch_X.dot(W1) + b1
                a1 = np.tanh(z1)
                z2 = a1.dot(W2) + b2
                a2 = np.tanh(z2)
                z3 = a2.dot(W3) + b3
                a3 = np.tanh(z3)
                z4 = a3.dot(W4) + b4
                exp_scores = np.exp(z4)
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

                # Backpropagation
                delta5 = probs
                delta5[range(batch_y.shape[0]), batch_y] -= 1
                dW4 = (a3.T).dot(delta5)
                db4 = np.sum(delta5, axis=0, keepdims=True)
                delta4 = np.multiply(delta5.dot(W4.T), 1 - np.power(a3, 2))
                dW3 = np.dot(a2.T, delta4)
                db3 = np.sum(delta4, axis=0)
                delta3 = np.multiply(delta4.dot(W3.T), 1 - np.power(a2, 2))
                dW2 = np.dot(a1.T, delta3)
                db2 = np.sum(delta3, axis=0)
                delta2 = np.multiply(delta3.dot(W2.T), 1 - np.power(a1, 2))
                dW1 = np.dot(batch_X.T, delta2)
                db1 = np.sum(delta2, axis=0)

                # Add regularization terms (b1 and b2 don't have regularization terms)
                dW4 += reg_lambda * W4
                dW3 += reg_lambda * W3
                dW2 += reg_lambda * W2
                dW1 += reg_lambda * W1

                # Gradient descent parameter update
                W1 = -epsilon * dW1 + W1
                b1 = -epsilon * db1 + b1
                W2 = -epsilon * dW2 + W2
                b2 = -epsilon * db2 + b2
                W3 = -epsilon * dW3 + W3
                b3 = -epsilon * db3 + b3
                W4 = -epsilon * dW4 + W4
                b4 = -epsilon * db4 + b4


                batch_start += batch_size

            if not epoch % 100:
                self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'] = W1, b1, W2, b2
                self.model['W3'], self.model['b3'], self.model['W4'], self.model['b4'] = W3, b3, W4, b4
                print(str(epoch) + " epoches loss: " +
                      str(self._calculate_loss(X, y, reg_lambda)))

if __name__ == "__main__":
    CLF = NNClassifier(input_features=784, hidden_features=196, output_features=10)
    vec_x, vec_y = fetch_openml('mnist_784', version=1, return_X_y=True)
    label_y = np.eye(10)[vec_y.astype(int)]
    Y_test = vec_y[60000:].astype(int)
    CLF.fit(vec_x[:60000], label_y[:60000], epsilon=0.001, num_passes=10000, reg_lambda=0.05)
    Y_predict = CLF.predict(vec_x[60000:]).astype(int)

    ncorrect = 0
    for dy in (Y_test - Y_predict):
        if 0 == dy:
            ncorrect += 1

    print('MNIST classification accuracy is {}%'.format(
        round(100.0*ncorrect/len(Y_test))))


