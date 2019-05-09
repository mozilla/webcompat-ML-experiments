import pandas as pd
import numpy as np
import json
import os
import csv
import sys
import warnings
from datetime import datetime
from math import floor
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb


def build_encoders(df):
    """Builds encoders for fields to be used when
    processing data for the model.

    All encoder specifications are stored in locally
    in /encoders as .json files.

    # Arguments
        df: A pandas DataFrame containing the data.
    """

    # Fit the Tokenizer.
    tokenizer = CountVectorizer(max_features=10000)
    tokenizer.fit(pd.concat([
        df['body'],
        df['title']
    ], axis=0).tolist())

    with open(os.path.join('encoders', 'model_vocab.json'),
              'w', encoding='utf8') as outfile:
        vocab = {k: int(v) for k, v in tokenizer.vocabulary_.items()}
        json.dump(vocab, outfile, ensure_ascii=False)

    # labels
    labels_tf = df['labels'].values
    labels_encoder = LabelBinarizer()
    labels_encoder.fit(labels_tf)

    with open(os.path.join('encoders', 'labels_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(labels_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Target Field: duplicate
    duplicate_encoder = LabelEncoder()
    duplicate_encoder.fit(df['duplicate'].values)

    with open(os.path.join('encoders', 'duplicate_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(duplicate_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)


def load_encoders():
    """Loads the encoders built during `build_encoders`.

    # Returns
        encoders: A dict of encoder objects/specs.
    """

    encoders = {}

    # Text
    tokenizer = CountVectorizer(max_features=10000)

    with open(os.path.join('encoders', 'model_vocab.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        tokenizer.vocabulary_ = json.load(infile)
    encoders['tokenizer'] = tokenizer

    # labels
    labels_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'labels_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        labels_encoder.classes_ = json.load(infile)
    encoders['labels_encoder'] = labels_encoder

    # Target Field: duplicate
    duplicate_encoder = LabelEncoder()

    with open(os.path.join('encoders', 'duplicate_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        duplicate_encoder.classes_ = np.array(json.load(infile))
    encoders['duplicate_encoder'] = duplicate_encoder

    return encoders


def process_data(df, encoders, process_target=True):
    """Processes an input DataFrame into a format
    sutable for model prediction.

    This function loads the encoder specifications created in
    `build_encoders`.

    # Arguments
        df: a DataFrame containing the source data
        encoders: a dict of encoders to process the data.
        process_target: boolean to determine if the target should be encoded.

    # Returns
        A tuple: A list containing all the processed fields to be fed
        into the model, and the processed target field.
    """

    # Transform and pad all text fields.

    # body
    body_enc = encoders['tokenizer'].transform(df['body'].values).toarray()

    # title
    title_enc = encoders['tokenizer'].transform(df['title'].values).toarray()

    # labels
    labels_enc = df['labels'].values
    labels_enc = encoders['labels_encoder'].transform(labels_enc)

    data_enc = [body_enc,
                labels_enc,
                title_enc
                ]

    if process_target:
        # Target Field: duplicate
        duplicate_enc = df['duplicate'].values

        duplicate_enc = encoders['duplicate_encoder'].transform(duplicate_enc)

        return (data_enc, duplicate_enc)

    return data_enc


def model_predict(df, model, encoders):
    """Generates predictions for a trained model.

    # Arguments
        df: A pandas DataFrame containing the source data.
        model: A compiled model.
        encoders: a dict of encoders to process the data.

    # Returns
        A numpy array of predictions.
    """

    data_enc = process_data(df, encoders, process_target=False)

    data_enc = xgb.DMatrix(np.hstack(data_enc))

    headers = ['probability']
    predictions = pd.DataFrame(model.predict(data_enc), columns=headers)

    return predictions


def model_train(df, encoders, args, model=None):
    """Trains a model, and saves the data locally.

    # Arguments
        df: A pandas DataFrame containing the source data.
        encoders: a dict of encoders to process the data.
        args: a dict of arguments passed through the command line
        model: A compiled model (for TensorFlow, None otherwise).
    """

    X, y_enc = process_data(df, encoders)
    X = np.hstack(X)
    y = df['duplicate'].values

    split = StratifiedShuffleSplit(
        n_splits=1, train_size=args.split, test_size=None, random_state=123)

    for train_indices, val_indices in split.split(np.zeros(y.shape[0]), y):
        train = xgb.DMatrix(X[train_indices, ], y[train_indices, ])
        val = xgb.DMatrix(X[val_indices, ], y[val_indices, ])

    params = {
        'eta': 0.1,
        'max_depth': 7,
        'gamma': 1,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_bin': 512,
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'silent': 1
    }

    f = open(os.path.join('metadata', 'results.csv'), 'w')
    w = csv.writer(f)
    w.writerow(['epoch', 'time_completed'] + ['log_loss',
                                              'accuracy', 'auc', 'precision', 'recall', 'f1'])

    y_true = y_enc[val_indices, ]
    for epoch in range(args.epochs):
        model = xgb.train(params, train, 1,
                          xgb_model=model if epoch > 0 else None)
        y_pred = model.predict(val)

        y_pred_label = np.round(y_pred)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logloss = log_loss(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred_label)
            precision = precision_score(y_true, y_pred_label, average='macro')
            recall = recall_score(y_true, y_pred_label, average='macro')
            f1 = f1_score(y_true, y_pred_label, average='macro')
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            auc_score = auc(fpr, tpr)

        metrics = [logloss,
                   acc,
                   auc_score,
                   precision,
                   recall,
                   f1]
        time_completed = "{:%Y-%m-%d %H:%M:%S}".format(datetime.utcnow())
        w.writerow([epoch+1, time_completed] + metrics)

        if args.context == 'automl-gs':
            sys.stdout.flush()
            print("\nEPOCH_END")

    f.close()
    model.save_model('model.bin')
