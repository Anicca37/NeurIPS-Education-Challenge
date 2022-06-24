import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import *


def split_data(data, m, n):
    """
    Split the given data by the threshold m and n into three pieces. Denote
    those pieces as Easy, Medium, Difficult, which represents the level of
    difficulty of the questions.
    :param data: Training data
    :param m: Lower threshold
    :param n: Upper threshold
    :return: Three data set that represents different level of the questions
    """
    student_id = data["user_id"]
    question_id = data["question_id"]
    is_correct = data["is_correct"]
    data = pd.DataFrame(np.array([student_id, question_id, is_correct]).T, columns=["user_id",
                                                                                    "question_id",
                                                                                    "is_correct"])
    # Compute difficulty level
    question_correct = data.groupby(["question_id"],
                                    sort=False
                                    )["is_correct"].sum().reset_index()
    question_num = data.groupby(["question_id"],
                                sort=False)['is_correct'].count().reset_index()
    question = question_num.merge(question_correct,
                                  on="question_id")
    question = question.rename(columns={"is_correct_x": 'num',
                                        "is_correct_y": "num_correct"})
    question["proportion"] = question["num_correct"] / question["num"]
    question = question.sort_values(by="proportion")
    print(question.quantile([x / 3 for x in range(4)]))

    question_easy = question.loc[question["proportion"] < m]
    data_easy = data[data["question_id"].isin(question_easy["question_id"])]
    question_medium = question.loc[(m <= question["proportion"])
                                   & (question["proportion"] < n)]
    data_medium = data[data["question_id"].isin(question_medium["question_id"])]
    question_difficult = question.loc[question["proportion"] >= n]
    data_difficult = data[data["question_id"].isin(question_difficult["question_id"])]

    easy_dict = {"question_id": data_easy["question_id"],
                 "user_id": data_easy["user_id"],
                 "is_correct": data_easy["is_correct"]}
    med_dict = {"question_id": data_medium["question_id"],
                "user_id": data_medium["user_id"],
                "is_correct": data_medium["is_correct"]}
    hard_dict = {"question_id": data_difficult["question_id"],
                 "user_id": data_difficult["user_id"],
                 "is_correct": data_difficult["is_correct"]}
    return easy_dict, med_dict, hard_dict


if __name__ == "__main__":
    train_data = load_train_csv("../data")
    question_easy, question_medium, question_difficult = split_data(train_data, 0.4, 0.6)
