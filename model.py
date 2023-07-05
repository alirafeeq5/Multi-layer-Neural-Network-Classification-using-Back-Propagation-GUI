import numpy as np

from util import ConfusionMatrix, get_logger, sigmoid


class Layer:

    def __init__(self, nodes_num, previous_num, bias=1, activation=sigmoid, eta=0.4):
        # Initialize a matrix N x M, where N is the number of nodes in the current layer,
        # and M is the number of nodes in the previous layer. Each row holds all the weights
        # connected to a node, and each column correspond to the weight of a node in the previous
        # layer that is connected to nodes in this layer. The bias is treated as a weight for each node.
        self.weights = np.random.standard_normal((nodes_num, previous_num + bias))

        self.activation = activation
        self.bias = bias
        self.eta = eta

        self.deltas = None # Hold the delta values calculated in the backward pass
        self.inputs = None # Hold the inputs recieved in the first forward pass
        self.nets = None # Hold the net values calculated in the first forward pass

    def get_outputs(self, inputs):
        # if bias is enabled, add 1 as a bias input
        if self.bias == 1:
            inputs = np.append(inputs, 1)

        # Reshape the inputs from 1D array to a 2D matrix
        self.inputs = np.array([inputs])

        # Calculate the net values using X W^T
        # (X W^T is different than W^T X since X is represented as a 1 x M matrix
        # instead of M x 1 matrix)
        self.nets = self.inputs @ self.weights.T

        # Return the activations of the net values as a 1D array
        return self.activation.function(self.nets).flatten()

    def calculate_deltas(self, back_deltas, back_weights):
        # Calculate the sums using matrix multiplication
        sums = back_deltas @ back_weights

        # If bias is enabled, add a bias output to match the dimensions of the sums matrix
        if self.bias == 1:
            self.nets = np.array([np.append(self.nets, 0)])

        # Calculate the delta values
        self.deltas = self.activation.derivative(self.nets) * sums

        # Because we calculate a delta for the bias weights in the current layer, that delta
        # must be removed since the previous layer doesn't provide outputs to the bias unit.
        forward_deltas = self.deltas
        if self.bias == 1:
            forward_deltas = np.array([self.deltas[0][:-1]])

        # Return both the delta values and weights so they can be used in the previuos layer
        return forward_deltas, self.weights

    def update_weights(self):
        # Convert the deltas and inputs matrices into an array
        self.deltas = self.deltas.flatten()
        self.inputs = self.inputs.flatten()

        # Update each row of weights according to the delta rule
        for i, row in enumerate(self.weights):
            row += self.eta * self.deltas[i] * self.inputs


class OutputLayer(Layer):

    def calculate_deltas(self, costs):
        # Output layers calculate the delta values based on the cost of the outputs
        self.deltas = costs * self.activation.derivative(self.nets)
        return self.deltas, self.weights


class MLP:
    
    def __init__(self, x_train, y_train, x_test, y_test, hidden_layers, bias=1, activation=sigmoid, eta=0.4, epochs=400, mse_threshold=0.05):
        # Store training and testing data
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        # The number of input nodes is the number of features (columns) in the training data
        self.num_inputs = len(x_train.columns)
        # The number of output nodes is the number of target unique values of the training data 
        self.num_outputs = len(y_train.unique())
        self.epochs = epochs
        self.mse_threshold = mse_threshold
        self.bias = bias

        # Create the hidden layers of the network
        self.layers = []
        prev = self.num_inputs
        for num in hidden_layers:
            self.layers.append(
                Layer(num, prev, bias, activation, eta)
            )
            prev = num

        # Create the output layer of the network
        self.layers.append(
            OutputLayer(self.num_outputs, prev, bias, activation, eta)
        )

        self.test_accuracy = -1
        self.train_accuracy = -1
        self.confusion_matrix = ConfusionMatrix(self.num_outputs)

        self.logger = get_logger(__name__ + "." + self.__class__.__name__)

        self.logger.info(f"MLP created with {hidden_layers} hidden layers, bias: {bias}, epochs: {epochs}, " +
                         f"activation: {activation.name}, learning rate: {eta}, MSE threshold: {mse_threshold}")

    def train(self):
        mses = []
        correct = 0

        for i in range(self.epochs):
            mse = 0

            for inputs, target in zip(self.x_train.values, self.y_train.values):

                # First forward pass to calculate the outputs
                ys = inputs
                for layer in self.layers:
                    ys = layer.get_outputs(ys)

                y = ys.argmax()
                correct += 1 if y == target else 0

                # Calculate the target array based on the value of the target. For example, if the target
                # is 2 out of a total of 6 classes (0-based), the target vector will be [0, 0, 1, 0, 0, 0]
                target_array = np.array(
                    [(1 if i == target else 0) for i in range(self.num_outputs)]
                )

                # Calculate the total cost to update the MSE
                total_cost = np.sum((ys - target_array) ** 2)
                mse += total_cost ** 2

                # Calculate the cost array for each output node (i.e: t - y for every node)
                costs = target_array - ys

                # Backward pass. Starts with the output layer, then goes backwards through the hidden layers
                deltas, weights = self.layers[-1].calculate_deltas(costs)
                for layer in reversed(self.layers[:-1]):
                    deltas, weights = layer.calculate_deltas(deltas, weights)

                # Second forward pass to update the weights
                for layer in self.layers:
                    layer.update_weights()

            # Calculate the MSE for the whole epoch and finish training if it's below the threshold
            mse *= 1 / len(self.y_train)

            mses.append(mse)

            self.logger.info(f"MSE at epoch {i + 1} is {mse}")

            if mse < self.mse_threshold:
                break

        self.train_accuracy = (correct / (len(mses) * len(self.y_train))) * 100

        self.logger.info(f"Finished training with accuracy {self.train_accuracy}")

        return mses, round(self.train_accuracy, 3)

    def test(self):
        correct = 0

        for inputs, target in zip(self.x_test.values, self.y_test.values):

            # Get the output of the network
            ys = inputs
            for layer in self.layers:
                ys = layer.get_outputs(ys)

            # Get the index of the maximum value, the index correspond to which class that output represents
            y = ys.argmax()

            self.confusion_matrix.add(target, y)

            # Determine of the network's choice is correct or not
            correct += 1 if y == target else 0

        # Calculate the accuracy
        self.test_accuracy = (correct / len(self.y_test)) * 100

        self.logger.info(f"Finished testing with accuracy {self.test_accuracy}")
        self.logger.info(f"The confusion matrix is:\n{self.confusion_matrix}")

        return self.confusion_matrix, round(self.test_accuracy, 3)

    def test_sample(self, sample):
        # Return the result of the network on a user-provided sample
        ys = sample
        for layer in self.layers:
            ys = layer.get_outputs(ys)

        return ys.argmax()
