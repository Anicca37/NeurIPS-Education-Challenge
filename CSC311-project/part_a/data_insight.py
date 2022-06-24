import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    student_data = pd.read_csv("../data/train_data.csv")

    total_num = student_data.shape[0]
    student_data["duplicated"] = student_data.duplicated(subset=['question_id', 'user_id'], keep=False)

    sum_duplicate = sum(student_data["duplicated"])

    # compute difficult level
    question_correct = student_data.groupby(["question_id"], sort=False)["is_correct"].sum().reset_index()
    question_num = student_data.groupby(["question_id"], sort=False)['is_correct'].count().reset_index()
    question = question_num.merge(question_correct, on="question_id")
    question = question.rename(columns={"is_correct_x": 'num', "is_correct_y": "num_correct"})
    question["proportion"] = question["num_correct"] / question["num"]
    question = question.sort_values(by="proportion")
    print(question)
    question_pro_easy = question.loc[question["proportion"] < 0.6]
    print(question_pro_easy['num'].sum())

    # plot accuracy rate of all questions
    plt.hist(question["proportion"], bins=20)
    plt.xlabel("Proportion")
    plt.ylabel("Number of Questions")
    plt.show()

    user_correct = student_data.groupby(["user_id"], sort=False)["is_correct"].sum().reset_index()
    user_num = student_data.groupby(["user_id"], sort=False)['is_correct'].count().reset_index()
    student = user_num.merge(user_correct, on="user_id")
    student = student.rename(columns={"is_correct_x": 'num', "is_correct_y": "num_correct"})
    print(student)
    student["proportion"] = student["num_correct"] / student["num"]

    # plt.hist(student["proportion"], bins=10)
    # plt.xlabel("Accuracy rate for question")
    # plt.ylabel("Number of Questions")
    # plt.show()
