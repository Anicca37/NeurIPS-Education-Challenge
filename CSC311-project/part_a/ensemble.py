# TODO: complete this file.
from utils import *
import numpy as np

from knn import *
from item_response import *
from neural_network import *
from part_a import item_response


def generate_matrix(matrix):
    """
    Generate one new sparse matrix using bootstrap.
    :param matrix: sparse training matrix
    :return: a new training matrix
    """
    sampled_matrix = np.ones((matrix.shape[0], matrix.shape[1]))

    n = matrix.shape[0]
    indices = np.random.choice(n, n, replace=True)  # generate a list of indices with replacement
    for i in range(n):
        idx = indices[i]
        sampled_matrix[i] = matrix[idx]
    return sampled_matrix


def generate_dataset(data):
    """
    Generate one new training dataset using bootstrap.
    :param data: training dataset
    :return: a new training dataset
    """
    sampled_data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }

    n = len(data["user_id"])
    indices = np.random.choice(n, n, replace=True)  # generate a list of indices with replacement
    for i in range(n):
        idx = indices[i]
        sampled_data["user_id"].append(data["user_id"][idx])
        sampled_data["question_id"].append(data["question_id"][idx])
        sampled_data["is_correct"].append(data["is_correct"][idx])

    return sampled_data


def knn_predictions(matrix, data):
    """
    Given matrix and data, predict the results using knn.
    :param matrix: training data matrix
    :param data: test data
    :return: predictions on data
    """
    predictions = []
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        pred = matrix[cur_user_id, cur_question_id]
        predictions.append(pred)
    return predictions


def nn_predictions(model, train_data, valid_data):
    """
    Given model, train_data and valid_data, predict the results using nn.
    :param model: neural network model
    :param train_data: training data
    :param valid_data: a dictionary
    :return: predictions on valid_data
    """
    predictions = []
    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)
        pred = output[0][valid_data["question_id"][i]].item()
        predictions.append(pred)
    return predictions


def irt_predictions(data, theta, beta):
    """
    Given data, theta and beta, predict the results using irt.
    :param data: a dictionary
    :param theta: vector
    :param beta: vector
    :return: predictions on data
    """
    predictions = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        predictions.append(p_a)
    return predictions


def ensemble_evaluate(data, predictions):
    """
    The evaluate function in utils.py.
    :param data: data to be evaluated
    :param predictions: list
    :return: accuracy
    """
    if len(data["is_correct"]) != len(predictions):
        raise Exception("Mismatch of dimensions between data and prediction.")
    if isinstance(predictions, list):
        predictions = np.array(predictions).astype(np.float64)
    return (np.sum((predictions >= 0.5) == data["is_correct"])
            / float(len(data["is_correct"])))


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    train_data = load_train_csv("../data")
    test_data = load_public_test_csv("../data")

    ###################
    # kNN with k = 11 #
    ###################
    # knn_matrix = generate_matrix(sparse_matrix)
    # nbrs = KNNImputer(n_neighbors=11)
    # knn_mat = nbrs.fit_transform(knn_matrix)
    #
    # knn_val_acc = knn_impute_by_user(knn_matrix, val_data, 11)
    # knn_test_acc = knn_impute_by_user(knn_matrix, test_data, 11)
    # print("knn validation dataset accuracy: {}, test dataset accuracy: {}".format(knn_val_acc, knn_test_acc))
    #
    # # predictions
    # knn_val_pred = knn_predictions(knn_mat, val_data)
    # knn_test_pred = knn_predictions(knn_mat, test_data)

    #####################################################################
    # neural network with k = 10, epoch = 20, lr = 0.1 and lamb = 0.001 #
    #####################################################################
    # nn_matrix = generate_matrix(sparse_matrix)
    # model = AutoEncoder(nn_matrix.shape[1], 10)
    # zero_train_matrix = nn_matrix.copy()
    # zero_train_matrix[np.isnan(nn_matrix)] = 0
    #
    # nn_matrix = torch.FloatTensor(nn_matrix)
    # zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    #
    # nn_val_acc = train(model, 0.1, 0.001, nn_matrix, zero_train_matrix, val_data, 20)[2]
    # nn_test_acc = train(model, 0.1, 0.001, nn_matrix, zero_train_matrix, test_data, 20)[2]
    # # print("neural network validation dataset accuracy: {}, test dataset accuracy: {}".format(nn_val_acc, nn_test_acc))
    #
    # # predictions
    # nn_val_pred = nn_predictions(model, zero_train_matrix, val_data)
    # nn_test_pred = nn_predictions(model, zero_train_matrix, test_data)

    ###########################################################
    # item response theory with iterations = 15 and lr = 0.01 #
    ###########################################################
    # irt_data = generate_dataset(train_data)
    # theta, beta, val_acc_lst = irt(irt_data, val_data, 0.01, 15)
    # irt_val_acc = item_response.evaluate(val_data, theta, beta)
    # irt_test_acc = item_response.evaluate(test_data, theta, beta)
    # print("item response theory validation dataset accuracy: {}, "
    #       "test dataset accuracy: {}".format(irt_val_acc, irt_test_acc))
    #
    # # predictions
    # irt_val_pred = irt_predictions(val_data, theta, beta)
    # irt_test_pred = irt_predictions(test_data, theta, beta)

    irt_data1 = generate_dataset(train_data)
    theta1, beta1, val_acc_lst1 = irt(irt_data1, val_data, 0.01, 15)
    # predictions
    irt_val_pred1 = irt_predictions(val_data, theta1, beta1)
    irt_test_pred1 = irt_predictions(test_data, theta1, beta1)

    irt_data2 = generate_dataset(train_data)
    theta2, beta2, val_acc_lst2 = irt(irt_data2, val_data, 0.01, 15)
    # predictions
    irt_val_pred2 = irt_predictions(val_data, theta2, beta2)
    irt_test_pred2 = irt_predictions(test_data, theta2, beta2)

    irt_data3 = generate_dataset(train_data)
    theta3, beta3, val_acc_lst3 = irt(irt_data3, val_data, 0.01, 15)
    # predictions
    irt_val_pred3 = irt_predictions(val_data, theta3, beta3)
    irt_test_pred3 = irt_predictions(test_data, theta3, beta3)

    ############
    # ensemble #
    ############
    val_len = len(val_data["is_correct"])
    ensemble_val_pred = []
    for i in range(val_len):
        # avg = (knn_val_pred[i] + nn_val_pred[i] + irt_val_pred[i]) / 3
        avg = (irt_val_pred1[i] + irt_val_pred2[i] + irt_val_pred3[i]) / 3
        ensemble_val_pred.append(avg)
    val_acc = ensemble_evaluate(val_data, ensemble_val_pred)
    print("Using bagging ensemble, the accuracy on validation data is: ", val_acc)

    test_len = len(test_data["is_correct"])
    ensemble_test_pred = []
    for j in range(test_len):
        # avg = (knn_test_pred[j] + nn_test_pred[j] + irt_test_pred[j]) / 3
        avg = (irt_test_pred1[j] + irt_test_pred2[j] + irt_test_pred3[j]) / 3
        ensemble_test_pred.append(avg)
    test_acc = ensemble_evaluate(test_data, ensemble_test_pred)
    print("Using bagging ensemble, the accuracy on test data is: ", test_acc)


if __name__ == "__main__":
    main()
