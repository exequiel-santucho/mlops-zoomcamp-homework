import pickle
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from prefect import flow, task

# MLflow setup
EXPERIMENT_NAME = "nyc-taxi-experiment"
# mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

# Ensure models folder exists
models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

@task
def read_dataframe(year: int, month: int) -> pd.DataFrame:
    """Read and preprocess trip data for a given year and month."""
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)
    print(f"Loaded records for {year}-{month:02d}: {len(df)}")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)
    print(f"Records after processing for {year}-{month:02d}: {len(df)}")
    return df

@task
def create_X(df: pd.DataFrame, dv: DictVectorizer = None) -> tuple:
    """Convert DataFrame to feature matrix using DictVectorizer."""
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

@task
def already_ran(client: MlflowClient, params: dict) -> bool:
    """Check if a run with the given parameters already exists."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.year = '{params['year']}' and params.month = '{params['month']}'",
        run_view_type=ViewType.ACTIVE_ONLY
    )
    return len(runs) > 0

@task
def train_and_log_model(X_train, y_train, X_val, y_val, dv: DictVectorizer, params: dict) -> str:
    """Train a linear regression model and log it to MLflow."""
    client = MlflowClient()
    if already_ran(client, params):
        print(f"Run for year={params['year']} month={params['month']} already exists. Skipping.")
        return None

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.sklearn.log_model(lr, artifact_path="model")
        print(f"Intercept of the model: {lr.intercept_}")

        return run.info.run_id

@task
def register_best_model() -> None:
    """Register the best model based on RMSE."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.rmse ASC"]
    )
    if not runs:
        print("No runs to register.")
        return
    best_run = runs[0]
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=EXPERIMENT_NAME)

@flow
def taxi_prediction_flow(year: int, month: int) -> str:
    """Main Prefect flow to orchestrate the taxi trip duration prediction pipeline."""
    # Read training and validation data
    df_train = read_dataframe(year, month)
    next_year, next_month = (year, month + 1) if month < 12 else (year + 1, 1)
    df_val = read_dataframe(next_year, next_month)

    # Create feature matrices
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    # Extract target variables
    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    # Train and log model
    params = {"year": str(year), "month": str(month)}
    run_id = train_and_log_model(X_train, y_train, X_val, y_val, dv, params)

    # Register best model if training was successful
    if run_id:
        print(f"MLflow run_id: {run_id}")
        register_best_model()
        with open("run_id.txt", "w") as f:
            f.write(run_id)
    return run_id

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    args = parser.parse_args()
    taxi_prediction_flow(args.year, args.month)