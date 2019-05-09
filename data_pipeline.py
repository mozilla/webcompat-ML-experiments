import json
import joblib
import numpy as np
import pandas as pd
import pathlib
import shutil

from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
WORKSPACE_PATH = pathlib.Path("workspace/{}".format(TIMESTAMP))


def create_workspace():
    print("Create workspace...")
    WORKSPACE_PATH.mkdir(parents=True, exist_ok=True)
    print("Copy raw dataset to workspace...")
    shutil.copy("dataset/initial_raw.csv", WORKSPACE_PATH)


def read_dataset():
    return pd.read_csv(WORKSPACE_PATH.joinpath("initial_raw.csv"))


def extract_categorical(df, columns=["labels", "invalid"]):
    for column in columns:
        le = preprocessing.LabelEncoder()
        df[column] = le.fit_transform(df[column])

        # Store label encoders
        persistent_path = WORKSPACE_PATH.joinpath(
            "{}_LabelEncoder.joblib".format(column)
        )
        joblib.dump(le, persistent_path)

    return df


def get_open_issues(df):
    return df[(df.state == "open")]


def get_closed_issues(df):
    return df[(df.state == "closed")]


def handle_empty_values(df):
    return df.fillna("EMPTY")


def extract_issue_labels(df):
    df["labels"] = df["labels"].apply(lambda labels: eval(labels))
    df["labels"] = df["labels"].apply(
        lambda labels: [label["name"] for label in labels]
    )
    df["labels"] = df["labels"].apply(lambda labels: sorted(labels))
    df["labels"] = df["labels"].apply(lambda labels: " ".join(labels))
    return df


def get_top_features(df, n, feature):
    counts = df[feature].value_counts()
    top = counts[:n]

    def _filter_count(row):
        return row[feature] in top

    return df[df.apply(_filter_count, axis=1)]


def extract_dataset(df, features=["body", "labels", "title", "invalid"]):
    return df[features]


def extract_invalid(df):
    df["invalid"] = df.apply(lambda x: "invalid" == x["milestone"], axis=1)
    return df


def train_validate_test_split(df, train_percent=0.6, validate_percent=0.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.ix[perm[:train_end]]
    validate = df.ix[perm[train_end:validate_end]]
    test = df.ix[perm[validate_end:]]
    return train, validate, test


def export_csv(df):
    train, validate, test = train_validate_test_split(df)
    df.to_csv(WORKSPACE_PATH.joinpath("dataset_full.csv"), index=False)
    train.to_csv(WORKSPACE_PATH.joinpath("dataset_train.csv"), index=False)
    validate.to_csv(WORKSPACE_PATH.joinpath("dataset_validate.csv"), index=False)
    test.to_csv(WORKSPACE_PATH.joinpath("dataset_test.csv"), index=False)


def pipeline():
    create_workspace()
    data = read_dataset()
    data = handle_empty_values(data)
    data = extract_issue_labels(data)
    data = get_top_features(data, 30, "labels")
    data = get_top_features(data, 10, "milestone")
    data = extract_invalid(data)
    data = extract_categorical(data)
    data = extract_dataset(data)
    export_csv(data)
    return data


if __name__ == "__main__":
    pipeline()
