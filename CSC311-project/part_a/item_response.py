from utils import *
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def evaluate(data, theta, beta, alpha):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """
    pred = []
    for i, q, r in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (alpha[r] * (theta[u] - beta[q])).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Turn data into sparse matrix
    user = np.array(data["user_id"])
    question = np.array(data["question_id"])
    is_correct = np.array(data["is_correct"])
    # sparseMatrix = csr_matrix((is_correct, (user, question)),
    #                           dtype=np.int8).toarray()
    num = user.shape[0]
    log_likelihood = 0
    for k in range(num):
        i = user[k]
        j = question[k]
        theta_i_minus_beta_j = theta[i] - beta[j]
        correct = is_correct[k]
        if correct == 1:
            log_likelihood += theta_i_minus_beta_j - np.log(1 + np.exp(theta_i_minus_beta_j))
        else:
            log_likelihood -= np.log(1 + np.exp(theta_i_minus_beta_j))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_likelihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    user = np.array(data["user_id"])
    question = np.array(data["question_id"])
    is_correct = np.array(data["is_correct"])

    d_theta = np.zeros(theta.shape)
    d_beta = np.zeros(beta.shape)
    num_data = user.shape[0]
    for k in range(num_data):
        i = user[k]
        j = question[k]
        is_correct_k = is_correct[k]

        expo = sigmoid(theta[i] - beta[j])
        if is_correct_k == 1:
            d_theta[i] += 1 - expo
            d_beta[j] += (-1) + expo
        elif is_correct_k == 0:  # answer == 0
            d_theta[i] -= expo
            d_beta[j] += expo
    theta = theta + lr * d_theta
    beta = beta + lr * d_beta

    is_correct[is_correct == 0] = 2
    sparseMatrix = csr_matrix((is_correct, (user, question)),
                              dtype=float).toarray()
    sparseMatrix[sparseMatrix == 0] = np.nan
    sparseMatrix[sparseMatrix == 2] = 0
    # a = 1
    # student_total_question = np.nansum(~np.isnan(sparseMatrix), axis=1)
    # question_total_student = np.nansum(~np.isnan(sparseMatrix), axis=0)
    # correct_answer_per_student = np.nansum(sparseMatrix, axis=1)
    # correct_student_per_answer = np.nansum(sparseMatrix, axis=0)
    # num_student, num_question = sparseMatrix.shape
    # # compute exp
    # theta_replicated = np.tile(theta, (num_question, 1))
    # beta_replicated = np.tile(beta, (num_student, 1)).T
    # exp_matrix = sigmoid(theta_replicated - beta_replicated)
    # # theta_grad = np.nansum(exp_matrix, axis=0) - (student_total_question - correct_answer_per_student)
    # theta_grad = correct_answer_per_student - np.sum(exp_matrix, axis=0)
    # theta = theta - lr * theta_grad
    # # beta_grad = (-1) * np.nansum(exp_matrix, axis=1) + question_total_student - correct_student_per_answer
    # beta_grad = (-1) * correct_student_per_answer + np.sum(exp_matrix, axis=1)
    # beta = beta - lr * beta_grad
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    user = np.array(data["user_id"])
    question = np.array(data["question_id"])
    is_correct = np.array(data["is_correct"])
    sparseMatrix = csr_matrix((is_correct, (user, question)),
                              dtype=np.int8).toarray()
    num_student, num_question = sparseMatrix.shape

    # theta = np.array([x for x in range(num_student)])
    theta = np.ones((num_student,), dtype=int)
    beta = np.zeros((num_question,), dtype=int)
    # beta = np.array([x for x in range(num_question)])

    val_acc_lst = []
    training_neg_lld_list = []
    val_neg_lld_list = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        training_neg_lld_list.append(-1 * neg_lld)
        valid_neg_lld = neg_log_likelihood(val_data, theta, beta)
        val_neg_lld_list.append(-1 * valid_neg_lld)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, val_acc_lst, training_neg_lld_list, val_neg_lld_list


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    iterations = 30
    lr = 0.02
    theta, beta, val_acc_lst, training_neg_lld_list, val_neg_lld_list = irt(train_data, val_data, lr, iterations)
    print(f"The learning rate is {lr}, and the number of iterations is {iterations}.")

    # test data
    val_accuracy = evaluate(val_data, theta, beta)
    test_accuracy = evaluate(test_data, theta, beta)
    print(f"The validation accuracy is {val_accuracy}")
    print(f"Test accuracy is {test_accuracy}.")

    # plot neg lld of training and validation as a function of iteration
    iteration_list = list(range(iterations))
    plt.plot(iteration_list, training_neg_lld_list, label="Train")
    plt.plot(iteration_list, val_neg_lld_list, label="Validation")
    plt.xlabel("Num of iterations")
    plt.ylabel("Log likelihood")
    plt.title("Log-likelihood on iterations of Training and Testing Data")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    theta_list = np.arange(-5, 5, 0.01)
    questions = [1525, 1574, 1030]
    beta_j = [beta[j] for j in questions]
    for j in range(len(questions)):
        prob = sigmoid(theta_list - beta_j[j])
        plt.plot(theta_list, prob, label=f"Question{questions[j]}")
    plt.title("Probability of correct response")
    plt.xlabel("Theta")
    plt.ylabel("Probability of Correctness")
    plt.legend()
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
