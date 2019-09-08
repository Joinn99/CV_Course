# Package imports
import numpy as np


class NNClassifier:
    # Initialize nn model
    def __init__(self, input_features=5, hidden_features=3, output_features=2):
        self.model = {"W1": np.random.random_sample((input_features, hidden_features)),
                      "b1": np.random.random_sample((1, hidden_features)),
                      "W2": np.random.random_sample((hidden_features, output_features)),
                      "b2": np.random.random_sample((1, output_features))}

    # Helper function to predict an output (0 or 1)
    def predict(self, x):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        # Forward propagation
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(np.array(probs), axis=1)

    def _calculate_loss(self, X, y, reg_lambda):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        # Forward propagation to calculate our predictions
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
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
                exp_scores = np.exp(z2)
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

                # Backpropagation
                delta3 = probs
                delta3[range(batch_y.shape[0]), batch_y] -= 1
                dW2 = (a1.T).dot(delta3)
                db2 = np.sum(delta3, axis=0, keepdims=True)
                delta2 = np.multiply(delta3.dot(W2.T), 1 - np.power(a1, 2))
                dW1 = np.dot(batch_X.T, delta2)
                db1 = np.sum(delta2, axis=0)

                # Add regularization terms (b1 and b2 don't have regularization terms)
                dW2 += reg_lambda * W2
                dW1 += reg_lambda * W1

                # Gradient descent parameter update
                W1 = -epsilon * dW1 + W1
                b1 = -epsilon * db1 + b1
                W2 = -epsilon * dW2 + W2
                b2 = -epsilon * db2 + b2

                batch_start += batch_size

            if not epoch % 100:
                self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'] = W1, b1, W2, b2
                print(str(epoch) + " epoches loss: " +
                      str(self._calculate_loss(X, y, reg_lambda)))
