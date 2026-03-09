"""
Adapted code from here: https://github.com/rasbt/machine-learning-book/blob/main/ch11/ch11.py
"""

import numpy as np
from sklearn.datasets import load_iris


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class NeuralNetMLP:
    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()

        self.num_classes = num_classes

        # hidden
        rng = np.random.RandomState(random_seed)

        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        # output
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        # Hidden layer
        # input dim: [n_examples, n_features] dot [n_hidden, n_features].T
        # output dim: [n_examples, n_hidden]
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # Output layer
        # input dim: [n_examples, n_hidden] dot [n_classes, n_hidden].T
        # output dim: [n_examples, n_classes]
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):

        #########################
        ### Output layer weights
        #########################

        # onehot encoding

        # Part 1: dLoss/dOutWeights
        ## = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeight
        ## where DeltaOut = dLoss/dOutAct * dOutAct/dOutNet
        ## for convenient re-use

        # BCE derivative * sigmoid derivative simplifies to (a_out - y):
        # d_loss/d_z_out = (-(y/a) + (1-y)/(1-a)) * a*(1-a) = a_out - y
        # output dim: [n_examples, n_classes]
        delta_out = (a_out - y) / y.shape[0]

        # gradient for output weights

        # [n_examples, n_hidden]
        d_z_out__dw_out = a_h

        # input dim: [n_classes, n_examples] dot [n_examples, n_hidden]
        # output dim: [n_classes, n_hidden]
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        #################################
        # Part 2: dLoss/dHiddenWeights
        ## = DeltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenNet * dHiddenNet/dWeight

        # [n_classes, n_hidden]
        d_z_out__a_h = self.weight_out

        # output dim: [n_examples, n_hidden]
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)

        # [n_examples, n_hidden]
        d_a_h__d_z_h = a_h * (1.0 - a_h)  # sigmoid derivative

        # [n_examples, n_features]
        d_z_h__d_w_h = x

        # output dim: [n_hidden, n_features]
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h)


def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx : start_idx + minibatch_size]

        yield X[batch_idx], y[batch_idx]


def binary_cross_entropy(y_true, y_pred):
    # Avoid log(0) - ensure that the log operates on valid inputs
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Compute Binary Cross-Entropy
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)


def evaluate_model_performance(nnet, X, y, minibatch_size=20):
    total_loss, correct_pred, num_examples, num_batches = 0.0, 0, 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)

    for i, (features, y_true) in enumerate(minibatch_gen):
        # forward pass = prediction = inference
        _, y_pred = nnet.forward(features)
        y_pred_labels = np.round(y_pred).astype(int)

        minibatch_loss = binary_cross_entropy(y_true, y_pred)
        correct_pred += (y_pred_labels == y_true).sum()

        num_examples += features.shape[0]
        total_loss += minibatch_loss
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    acc = correct_pred / num_examples
    return avg_loss, acc


def train(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    num_epochs,
    learning_rate=0.1,
    minibatch_size=20,
):

    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for e in range(num_epochs):
        # iterate over minibatches
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)

        for X_train_mini, y_train_mini in minibatch_gen:
            #### Compute outputs ####
            a_h, a_out = model.forward(X_train_mini)

            #### Compute gradients ####
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = (
                model.backward(X_train_mini, a_h, a_out, y_train_mini)
            )

            #### Update weights ####
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out

        #### Epoch Logging ####
        train_loss, train_acc = evaluate_model_performance(model, X_train, y_train)
        valid_loss, valid_acc = evaluate_model_performance(model, X_valid, y_valid)
        train_acc, valid_acc = train_acc * 100, valid_acc * 100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_loss)
        print(
            f"Epoch: {e + 1:03d}/{num_epochs:03d} "
            f"| Train BCE: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.2f}% "
            f"| Valid Acc: {valid_acc:.2f}%"
        )

    return epoch_loss, epoch_train_acc, epoch_valid_acc


def load_data():
    """
    Modify the iris dataset for binary classification (predicting 'setosa' only),
    and select only the sepal dimensions (first two features).
    """
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]

    # Convert to binary classification: predict only 'setosa' (target 0)
    y = (y == 0).astype(int)
    y = y.reshape(-1, 1)

    # Select only sepal dimensions (first two features)
    X = X[:, :2]

    return X, y


def main():
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_data()

    # Split data

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=20, random_state=123, stratify=y
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=20, random_state=123, stratify=y_temp
    )

    # Create model
    model = NeuralNetMLP(num_features=2, num_hidden=10, num_classes=1)

    # Performance with initial weights
    avg_loss, acc = evaluate_model_performance(model, X_train, y_train)
    print(f"Initial BCE loss: {avg_loss:.4f}")
    print(f"Initial accuracy: {acc * 100:.1f}%")

    # Train
    np.random.seed(123)  # for the training set shuffling
    epoch_loss, epoch_train_acc, epoch_valid_acc = train(
        model, X_train, y_train, X_valid, y_valid, num_epochs=50, learning_rate=0.2
    )


if __name__ == "__main__":
    main()
