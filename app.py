import json
import os
from hashlib import md5
from time import localtime

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from joblib import load, dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

app = Flask(__name__)

uploads_dir = os.path.join(app.root_path, 'csv')
model_rfc_path = os.path.join(app.root_path, 'models', 'rfc')
columns_path = os.path.join(app.root_path, 'models', 'columns.json')

@app.errorhandler(Exception)
def error_handler(err):
    response = jsonify(str(err))
    response.status_code = 400
    return response

@app.route('/csv', methods=['POST'])
def csv():
    if 'file' not in request.files:
        raise Exception({"error": "No file was provided"})

    file = request.files['file']
    prefix = md5(str(localtime()).encode('utf-8')).hexdigest()
    path = os.path.join(uploads_dir, f"{prefix}_{file.filename}")
    file.save(path)

    response = jsonify(success=True)
    response.status_code = 200
    return response


@app.route('/model/train', methods=['PUT'])
def train_model():
    files = []

    for filename in os.listdir(uploads_dir):
        if filename == '.gitkeep':
            continue
        path = os.path.join(uploads_dir, filename)
        df = pd.read_csv(path, index_col=None, header=0)
        files.append(df)

    df = pd.concat(files, axis=0, ignore_index=True)

    clean_dataframe(df)

    df_no_transaction_type = df.drop('transactionType', axis=1)
    X = pd.get_dummies(df_no_transaction_type)
    y = df['transactionType']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train, y_train)

    rfc_pred = rfc.predict(X_test)
    classificationReport = classification_report(y_test, rfc_pred, output_dict=True)
    confusionMatrix = confusion_matrix(y_test, rfc_pred)

    dump(rfc, open(model_rfc_path, 'wb+'))

    columns = X_train.columns.values
    with open(columns_path, 'w+') as fout:
        fout.write(json.dumps(columns.tolist()))

    result = {
        "classificationReport": classificationReport,
        "confusionMatrix": confusionMatrix.tolist()
    }

    return jsonify(result)


@app.route('/model/predict', methods=['POST'])
def predict():
    payload = json.loads(request.data)

    rfc = load(model_rfc_path)

    with open(columns_path) as json_file:
        required_columns = json.load(json_file)

    data = pd.get_dummies(pd.DataFrame(payload if isinstance(payload, list) else [payload]))
    already_added_columns = data.columns.values

    for missing_column in required_columns:
        if missing_column in already_added_columns:
            continue
        data[missing_column] = 0

    result = rfc.predict(data).tolist()

    return jsonify({
        "results": result
    })


def clean_dataframe(df):
    df.drop(['ID'], axis=1, inplace=True)
    df.drop(['timestamp'], axis=1, inplace=True)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    df['elapsedTime'].replace('', np.nan, inplace=True)
    df['transactionAmount'].replace('', np.nan, inplace=True)

    df[['elapsedTime']] = imputer.fit_transform(df[['elapsedTime']])
    df[['transactionAmount']] = imputer.fit_transform(df[['transactionAmount']])

    df['countryOrigin'].replace('', np.nan, inplace=True)
    df['transactionType'].replace('', np.nan, inplace=True)

    df.dropna(subset=['countryOrigin'], inplace=True)
    df.dropna(subset=['transactionType'], inplace=True)
