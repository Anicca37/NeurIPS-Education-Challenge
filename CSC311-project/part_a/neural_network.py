from matplotlib import pyplot as plt

from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = torch.sigmoid(self.g(inputs))
        out = torch.sigmoid(self.h(out))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    training_loss = []
    val_accuracy = []
    num_iteration = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            regularizer = 0.5 * lamb * model.get_weight_norm()
            loss = torch.sum((output - target) ** 2.) + regularizer
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        training_loss.append(train_loss)
        num_iteration.append(epoch + 1)

        valid_acc = evaluate(model, zero_train_data, valid_data)
        val_accuracy.append(valid_acc)
        # print("Epoch: {} \tTraining Cost: {:.6f}\t "
        #       "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Test Acc: {}".format(epoch, train_loss, valid_acc))

    return num_iteration, training_loss, val_accuracy
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    num_questions = train_matrix.shape[1]

    k_list = [10, 50, 100, 200, 500]
    learning_rates = [0.1, 0.001]
    iteration_list = [20, 50]
    regularization_penalty = [0.001, 0.01, 0.1, 1]
    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k = k_list[0]
    model = AutoEncoder(num_questions, k)

    # Set optimization hyperparameters.
    lr = learning_rates[0]
    num_epoch = iteration_list[0]
    lamb = regularization_penalty[0]

    num_iteration, training_loss, val_accuracy = train(model, lr, lamb,
                                                       train_matrix,
                                                       zero_train_matrix,
                                                       valid_data, num_epoch)

    # num_iteration, training_loss, test_accuracy = train(model, lr, lamb,
    #                                                    train_matrix,
    #                                                    zero_train_matrix,
    #                                                    test_data, num_epoch)

    # Plot the graph of training loss along with number of epochs
    plt.plot(num_iteration, training_loss, label='Training Loss')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.show()

    # Plot the graph of validation accuracy with number of epochs
    plt.plot(num_iteration, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.show()

    # # # Plot the graph of Test accuracy with number of epochs
    # plt.plot(num_iteration, test_accuracy, label='Test Accuracy')
    # plt.xlabel('Number of Epochs')
    # plt.ylabel('Test Accuracy')
    # plt.legend()
    # plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
