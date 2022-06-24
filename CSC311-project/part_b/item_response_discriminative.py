from utils import *
from classification import *
from scipy.sparse import csr_matrix


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, alpha):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param alpha: discrimination parameter
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
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
        alpha_theta_i_minus_beta_j = alpha[j] * (theta[i] - beta[j])
        correct = is_correct[k]
        if correct == 1:
            log_likelihood += alpha_theta_i_minus_beta_j - np.log(1 + np.exp(alpha_theta_i_minus_beta_j))
        else:
            log_likelihood -= np.log(1 + np.exp(alpha_theta_i_minus_beta_j))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_likelihood


def evaluate_0(data, theta, beta, alpha):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (alpha[q] * (theta[u] - beta[q])).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def update_theta_beta_alpha(data, lr, theta, beta, alpha):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param alpha: discrimination parameter
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
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
    d_alpha = np.zeros(alpha.shape)

    num_data = user.shape[0]
    for k in range(num_data):
        i = user[k]
        j = question[k]
        is_correct_k = is_correct[k]

        expo = sigmoid(alpha[j] * (theta[i] - beta[j]))
        if is_correct_k == 1:
            d_theta[i] += alpha[j] * (1 - expo)
            d_beta[j] += alpha[j] * ((-1) + expo)
            d_alpha[j] += (theta[i] - beta[j]) * (1 - expo)
        elif is_correct_k == 0:  # answer == 0
            d_theta[i] -= alpha[j] * expo
            d_beta[j] += alpha[j] * expo
            d_alpha[j] -= expo * (theta[i] - beta[j])
    theta = theta + lr * d_theta
    beta = beta + lr * d_beta
    alpha = alpha + lr * d_alpha

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, alpha


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, alpha, val_acc_lst, training_neg_lld_list, val_neg_lld_list)
    """
    user = np.array(data["user_id"])
    question = np.array(data["question_id"])
    is_correct = np.array(data["is_correct"])
    # sparseMatrix = csr_matrix((is_correct, (user, question)),
    #                           dtype=np.int8).toarray()
    num_student = 542
    num_question = 1774

    theta = np.ones((num_student,), dtype=int)
    beta = np.zeros((num_question,), dtype=int)
    alpha = np.zeros((num_question,), dtype=int)
    # theta = np.array([x for x in range(num_student)])
    # beta = np.array([x for x in range(num_question)])

    val_acc_lst = []
    training_neg_lld_list = []
    val_neg_lld_list = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha)
        training_neg_lld_list.append(-1 * neg_lld)
        valid_neg_lld = neg_log_likelihood(val_data, theta, beta, alpha)
        val_neg_lld_list.append(-1 * valid_neg_lld)
        score = evaluate_0(data=val_data, theta=theta, beta=beta, alpha=alpha)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, alpha = update_theta_beta_alpha(data, lr, theta, beta, alpha)

    return theta, beta, alpha, training_neg_lld_list, question


def evaluate(data, theta_0, theta_1, theta_2, beta_split, alpha_split, hardness):
    """ Evaluate the model given data and return the accuracy.
    :param theta_0: thetas for easy
    :param theta_1: thetas for medium
    :param theta_2: thetas for hard
    :param beta_split: the merged betas
    :param alpha_split: the merged alpha
    :param hardness: the hardness of questions
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        # evaluate which list is this question in
        u = data["user_id"][i]
        # 3 cases according to difficulty level
        if hardness[q] == 0:  # easy
            theta_i = theta_0[u]
        elif hardness[q] == 1:  # medium
            theta_i = theta_1[u]
        else:  # hard
            theta_i = theta_2[u]
        x = (alpha_split[q] * (theta_i - beta_split[q])).sum()
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
    iterations = 20
    lr = 0.01
    iterations_0 = 40
    lr_0 = 0.03
    iterations_1 = 50
    lr_1 = 0.01
    iterations_2 = 20
    lr_2 = 0.01
    low_percentile = 0.53
    higher_percentile = 0.68

    data_easy, data_medium, data_difficult = split_data(train_data, low_percentile, higher_percentile)
    print("Starting whole data\n")
    theta, beta, alpha, training_neg_lld_list_0, question_id_list_0 = \
        irt(train_data, val_data, lr, iterations)
    print("Starting easy data 0 \n")
    theta_0, beta_0, alpha_0, training_neg_lld_list_0, question_id_list_0 = \
        irt(data_easy, val_data, lr_0, iterations_0)
    print("Starting mid data 1 \n")
    theta_1, beta_1, alpha_1, training_neg_lld_list_1, question_id_list_1 = \
        irt(data_medium, val_data, lr_1, iterations_1)
    print("Starting hard data 2 \n")
    theta_2, beta_2, alpha_2, training_neg_lld_list_2, question_id_list_2 = \
        irt(data_difficult, val_data, lr_2, iterations_2)

    # Next, I want to merge beta and alpha
    beta_split = []
    alpha_split = []
    hardness = np.empty(beta.shape[0])
    for k in range(beta.shape[0]):
        if beta_0[k] != 0:
            beta_split.append(beta_0[k])
            alpha_split.append(alpha_0[k])
            hardness[k] = 0
        elif beta_1[k] != 0:
            beta_split.append(beta_1[k])
            alpha_split.append(alpha_1[k])
            hardness[k] = 1
        else:
            beta_split.append(beta_2[k])
            alpha_split.append(alpha_2[k])
            hardness[k] = 2
    beta_split = np.array(beta_split)
    alpha_split = np.array(alpha_split)

    # Fill in values where theta is 0
    for k in range(theta.shape[0]):
        if theta_0[k] == 0:
            print("Not full")
            theta_0[k] = theta[k]
        if theta_1[k] == 0:
            print("Not full")
            theta_1[k] = theta[k]
        if theta_2[k] == 0:
            print("Not full")
            theta_2[k] = theta[k]

    print(f"The learning rate is {lr}, and the number of iterations is {iterations}.")

    # test data, predict and evaluate value
    val_accuracy = evaluate(val_data, theta_0, theta_1, theta_2, beta_split, alpha_split, hardness)
    test_accuracy = evaluate(test_data, theta_0, theta_1, theta_2, beta_split, alpha_split, hardness)
    print(f"The validation accuracy for easy questions is {val_accuracy}")
    print(f"Test accuracy is {test_accuracy}.")

    # plot neg lld of training and validation as a function of iteration
    # iteration_list = list(range(iterations))
    # plt.plot(iteration_list, training_neg_lld_list, label="Train")
    # plt.plot(iteration_list, val_neg_lld_list, label="Validation")
    # plt.xlabel("Num of iterations")
    # plt.ylabel("Log likelihood")
    # plt.title("Log-likelihood on iterations of Training and Testing Data")
    # plt.legend()
    # plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # # Implement part (d)                                              #
    # ###################################################################
    # theta_list = np.arange(-5, 5, 0.01)
    # questions = [1525, 1574, 1030]
    # beta_j = [beta[j] for j in questions]
    # for j in range(len(questions)):
    #     prob = sigmoid(theta_list - beta_j[j])
    #     plt.plot(theta_list, prob, label=f"Question{questions[j]}")
    # plt.title("Probability of correct response")
    # plt.xlabel("Theta")
    # plt.ylabel("Probability of Correctness")
    # plt.legend()
    # plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
