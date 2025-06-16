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
mlflow.set_tracking_uri("sqlite:///mlflow.db") # take data from db
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
    
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
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
def train_and_log_model(X_train, y_train, X_val, y_val, dv: DictVectorizer) -> str:
    """Train a linear regression model and log it to MLflow."""
    with mlflow.start_run() as run:
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
def register_model(X_train, y_train, X_val, y_val, dv: DictVectorizer, top_n: int) -> None:
    """Register the top model based on RMSE."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(X_train, y_train, X_val, y_val, dv)

    if runs:
        best_run = runs[0]
        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=EXPERIMENT_NAME)
    else:
        print("No runs found to register.")

@flow
def taxi_prediction_flow(year: int, month: int) -> str:
    """Main Prefect flow to orchestrate the taxi trip duration prediction pipeline."""
    # Read training and validation data
    df_train = read_dataframe(year, month)
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(next_year, next_month)

    # Create feature matrices
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    # Extract target variables
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    # Train and log model
    run_id = train_and_log_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")

    # Register model
    register_model(X_train, y_train, X_val, y_val, dv, top_n=1)

    # Write run_id to file
    with open("run_id.txt", "w") as f:
        f.write(run_id)

    return run_id

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()
    taxi_prediction_flow(year=args.year, month=args.month)