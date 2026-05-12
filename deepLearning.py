import numpy as np

class DeepLearning:
    """
    This class defines the vanilla optimization of a deep learning model.
    Args:
        layer_of_units (list): A list to specify the number of units in each layer.
    """
    def __init__(self, layer_of_units):
        self._n_layers = len(layer_of_units)
        parameters = {}
        for i in range(self._n_layers - 1):
            parameters['W{}'.format(i + 1)] = np.random.rand(layer_of_units[i + 1], layer_of_units[i])
            parameters['B{}'.format(i + 1)] = np.random.rand(layer_of_units[i + 1], 1)
        self._parameters = parameters
    def sigmoid(self, Z):
        """
        This function returns the Sigmoid output.
        Args:
            Z (ndarray): The multiplication of weights and output from previous layer.
        """
        return 1/(1 + np.exp(-Z))
    def single_layer_forward_propagation(self, A_previous, W_current, B_current):
        """
        This function returns the output of a single layer of forward propagation.
        Args:
            A_previous (ndarray): The Sigmoid output from previous layer.
            W_current (ndarray): The weights of current layer.
            B_current (ndarray): The bias of current layer.
        """
        Z_current = np.dot(W_current, A_previous) + B_current
        A_current = self.sigmoid(Z_current)
        return A_current, Z_current
    def forward_propagation(self):
        """
        This function returns the output of a complete round of forward propagation.
        """
        self._m = self._X_train.shape[0]
        X_train_T = self._X_train.copy().T
        cache = {}
        A_current = X_train_T
        for i in range(self._n_layers - 1):
            A_previous = A_current
            W_current = self._parameters["W{}".format(i + 1)]
            B_current = self._parameters["B{}".format(i + 1)]
            A_current, Z_current = self.single_layer_forward_propagation(A_previous, W_current, B_current)
            cache["A{}".format(i)] = A_previous
            cache["Z{}".format(i + 1)] = Z_current
        self._cache = cache
        self._A_current = A_current
    def derivative_sigmoid(self, Z):
        """
        This function returns the output of the derivative of Sigmoid function.
        Args:
            Z (ndarray): The multiplication of weights, bias and output from previous layer.
        """
        sig = self.sigmoid(Z)
        return sig * (1 - sig)
    def single_layer_backward_propagation(self, dA_current, W_current, B_current, Z_current, A_previous):
        """
        This function returns the output of a single layer of backward propagation.
        Args:
            dA_current (ndarray): The output of the derivative of Sigmoid function from previous layer.
            W_current (ndarray): The weights of current layer.
            B_current (ndarray): The bias of current layer.
            Z_current (ndarray): The multiplication of weights, bias and output from previous layer.
            A_previous (ndarray): The Sigmoid output from previous layer.
        """
        dZ_current = dA_current * self.derivative_sigmoid(Z_current)
        dW_current = np.dot(dZ_current, A_previous.T) / self._m
        dB_current = np.sum(dZ_current, axis=1, keepdims=True) / self._m
        dA_previous = np.dot(W_current.T, dZ_current)
        return dA_previous, dW_current, dB_current
    def backward_propagation(self):
        """
        This function performs a complete round of backward propagation to update weights and bias.
        """
        gradients = {}
        self.forward_propagation()
        Y_hat = self._A_current.copy()
        Y_train = self._y_train.copy().reshape(1, self._m)
        dA_previous = - (np.divide(Y_train, Y_hat) - np.divide(1 - Y_train, 1 - Y_hat))
        for i in reversed(range(dl._n_layers - 1)):
            dA_current = dA_previous
            A_previous = self._cache["A{}".format(i)]
            Z_current = self._cache["Z{}".format(i+1)]
            W_current = self._parameters["W{}".format(i+1)]
            B_current = self._parameters["B{}".format(i+1)]
            dA_previous, dW_current, dB_current = self.single_layer_backward_propagation(dA_current, W_current, B_current, Z_current, A_previous)
            gradients["dW{}".format(i + 1)] = dW_current
            gradients["dB{}".format(i + 1)] = dB_current
        self._gradients = gradients
    def cross_entropy(self):
        """
        This function returns the cross entropy given weights and bias.
        """
        Y_hat = self._A_current.copy()
        self._Y_hat = Y_hat
        Y_train = self._y_train.copy().reshape(1, self._m)
        ce = -1 / self._m * (np.dot(Y_train, np.log(Y_hat).T) + np.dot(1 - Y_train, np.log(1 - Y_hat).T))
        return ce[0, 0]
    def accuracy_score(self):
        """
        This function returns the accuracy score given weights and bias.
        """
        p_pred = self._Y_hat.ravel()
        y_pred = np.where(p_pred > 0.5, 1, 0)
        y_true = self._y_train
        accuracy = (y_pred == y_true).sum() / y_pred.size
        return accuracy
    def gradient_descent(self):
        """
        This function performs vanilla gradient descent to update weights and bias.
        """
        for i in range(self._n_layers - 1):
            self._parameters["W{}".format(i + 1)] -= self._learning_rate * self._gradients["dW{}".format(i + 1)]
            self._parameters["B{}".format(i + 1)] -= self._learning_rate * self._gradients["dB{}".format(i + 1)]
    def fit(self, X_train, y_train, epochs=100000, learning_rate=0.001):
        """
        This function uses multiple rounds of forward propagations and backward propagations to optimize weights and bias.
        Args:
            X_train (ndarray): 2d-array for feature matrix of training data.
            y_train (ndarray): 1d-array for target vector of training data.
            epochs (int): The number of iterations to update the model weights.
            learning_rate (float): The learning rate of gradient descent.
        """
        self._X_train = X_train.copy()
        self._y_train = y_train.copy()
        self._learning_rate = learning_rate
        loss_history = []
        accuracy_history = []
        n_prints = 10
        print_iter = epochs // n_prints
        for i in range(epochs):
            self.forward_propagation()
            ce = self.cross_entropy()
            accuracy = self.accuracy_score()
            loss_history.append(ce)
            accuracy_history.append(accuracy)
            self.backward_propagation()
            self.gradient_descent()
            if i % print_iter == 0:
                print("Iteration: {:6} - cost: {:.6f} - accuracy: {:.2f}%".format(i, ce, accuracy * 100))
        self._loss_history = loss_history
        self._accuracy_history = accuracy_history
    def predict_proba(self, X_test):
        """
        This function returns predicted probability for class 1 with weights of this model.
        Args:
            X_test (ndarray): 2d-array for feature matrix of test data.
        """
        X_test_T = X_test.copy().T
        A_current = X_test_T
        for i in range(self._n_layers - 1):
            A_previous = A_current
            W_current = self._parameters["W{}".format(i + 1)]
            B_current = self._parameters["B{}".format(i + 1)]
            A_current, Z_current = self.single_layer_forward_propagation(A_previous, W_current, B_current)
            self._cache["A{}".format(i)] = A_previous
            self._cache["Z{}".format(i + 1)] = Z_current
        p_hat_1 = A_current.copy().ravel()
        return p_hat_1
    def predict(self, X_test):
        p_hat_1 = self.predict_proba(X_test)
        return np.where(p_hat_1 >= 0.5, 1, 0)
