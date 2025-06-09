from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# MLflow setup
EXPERIMENT_NAME = "nyc-taxi-experiment"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

# Ensure models folder exists
models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

# Task 1: Read and preprocess data
def read_and_preprocess_data(year, month, **context):
    def read_dataframe(year, month):
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
        df = pd.read_parquet(url)
        print(f"Loaded records for {year}-{month}: {len(df)}")
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)
        print(f"Records after processing for {year}-{month}: {len(df)}")
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

    # Read training and validation data
    df_train = read_dataframe(year, month)
    next_year, next_month = (year, month + 1) if month < 12 else (year + 1, 1)
    df_val = read_dataframe(next_year, next_month)

    # Create features
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    # Push data to XCom for downstream tasks
    context['ti'].xcom_push(key='X_train', value=X_train)
    context['ti'].xcom_push(key='y_train', value=df_train['duration'].values)
    context['ti'].xcom_push(key='X_val', value=X_val)
    context['ti'].xcom_push(key='y_val', value=df_val['duration'].values)
    context['ti'].xcom_push(key='dv', value=dv)

# Task 2: Train and log model
def train_and_log_model(**context):
    client = MlflowClient()
    params = context['dag_run'].conf.get('params', {'year': '2023', 'month': '1'})
    year = int(params['year'])
    month = int(params['month'])

    def already_ran(client, params):
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"params.year = '{params['year']}' and params.month = '{params['month']}'",
            run_view_type=ViewType.ACTIVE_ONLY
        )
        return len(runs) > 0

    if already_ran(client, params):
        print(f"Run for year={year} month={month} already exists. Skipping.")
        return None

    # Pull data from XCom
    X_train = context['ti'].xcom_pull(key='X_train')
    y_train = context['ti'].xcom_pull(key='y_train')
    X_val = context['ti'].xcom_pull(key='X_val')
    y_val = context['ti'].xcom_pull(key='y_val')
    dv = context['ti'].xcom_pull(key='dv')

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

        context['ti'].xcom_push(key='run_id', value=run.info.run_id)
        return run.info.run_id

# Task 3: Register best model
def register_best_model(**context):
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

# Task 4: Save run ID
def save_run_id(**context):
    run_id = context['ti'].xcom_pull(key='run_id')
    if run_id:
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        print(f"Saved run_id: {run_id}")

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'mlops_taxi_dag',
    default_args=default_args,
    description='MLOps pipeline for NYC taxi trip duration prediction',
    schedule_interval=None,  # Run manually or set to e.g., '@monthly'
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    preprocess_task = PythonOperator(
        task_id='read_and_preprocess_data',
        python_callable=read_and_preprocess_data,
        op_kwargs={'year': '{{ dag_run.conf["year"] }}', 'month': '{{ dag_run.conf["month"] }}'},
    )

    train_task = PythonOperator(
        task_id='train_and_log_model',
        python_callable=train_and_log_model,
    )

    register_task = PythonOperator(
        task_id='register_best_model',
        python_callable=register_best_model,
    )

    save_run_task = PythonOperator(
        task_id='save_run_id',
        python_callable=save_run_id,
    )

    # Define task dependencies
    preprocess_task >> train_task >> register_task >> save_run_task