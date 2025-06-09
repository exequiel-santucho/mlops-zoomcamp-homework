#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

EXPERIMENT_NAME = "nyc-taxi-experiment"
mlflow.set_tracking_uri("http://localhost:5000") # first we need to create db in console: mlflow ui --backend-store-uri sqlite:///mlflow.db
mlflow.set_experiment(EXPERIMENT_NAME)

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


def read_dataframe(year, month):

    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    print("Loaded records: ", len(df))

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print("Records after processing: ", len(df))
    
    return df


def create_X(df, dv=None):

    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_and_log_model(X_train, y_train, X_val, y_val, dv):

    with mlflow.start_run() as run:

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        # Log the model
        mlflow.sklearn.log_model(lr, artifact_path="model")

        # Show intercept of the model
        print(f"Intercept of the model: {lr.intercept_}")

        return run.info.run_id


def register_model(X_train, y_train, X_val, y_val, dv, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(X_train, y_train, X_val, y_val, dv)

    best_run = runs[0]

    # Register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=EXPERIMENT_NAME)


def run(year, month):

    df_train = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_and_log_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")

    register_model(X_train, y_train, X_val, y_val, dv, top_n=1)

    return run_id


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)

    with open("run_id.txt", "w") as f:
        f.write(run_id)