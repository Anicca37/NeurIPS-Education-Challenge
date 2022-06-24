from sklearn.impute import KNNImputer
from utils import *

import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_vals = [1, 6, 11, 16, 21, 26]

    ######################################
    # user-based collaborative filtering #
    ######################################
    acc_user = []

    # compute the accuracy on validation data
    for k in k_vals:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        acc_user.append(acc)

    # plot and report
    plt.plot(k_vals, acc_user)
    plt.xticks(k_vals)
    plt.xlabel("values of k")
    plt.ylabel("accuracy")
    plt.title("kNN using user-based collaborative filtering")
    plt.show()

    # run on test data with highest performance
    highest_idx1 = acc_user.index(max(acc_user))
    highest_k1 = k_vals[highest_idx1]
    highest_acc1 = knn_impute_by_user(sparse_matrix, test_data, highest_k1)
    print("The final test accuracy is: {}% with k* = {}.\n".format(highest_acc1 * 100, highest_k1))

    ######################################
    # item-based collaborative filtering #
    ######################################
    acc_item = []

    # compute the accuracy on validation data
    for k in k_vals:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        acc_item.append(acc)

    # plot and report
    plt.plot(k_vals, acc_item)
    plt.xticks(k_vals)
    plt.xlabel("values of k")
    plt.ylabel("accuracy")
    plt.title("kNN using item-based collaborative filtering")
    plt.show()

    # run on test data with highest performance
    highest_idx2 = acc_item.index(max(acc_item))
    highest_k2 = k_vals[highest_idx2]
    highest_acc2 = knn_impute_by_item(sparse_matrix, test_data, highest_k2)
    print("The final test accuracy is: {}% with k* = {}.\n".format(highest_acc2 * 100, highest_k2))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
