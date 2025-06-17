#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")

def load_model(model_path: str) -> tuple[DictVectorizer, LinearRegression]:
    """Load the pre-trained DictVectorizer and LinearRegression model from a pickle file."""
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def read_data(filename: str, categorical: list[str]) -> pd.DataFrame:
    """Read and preprocess trip data from a Parquet file."""
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

def predict(df: pd.DataFrame, dv: DictVectorizer, model: LinearRegression, categorical: list[str]) -> np.ndarray:
    """Generate predictions using the provided model and data."""
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred

def create_result_dataframe(df: pd.DataFrame, y_pred: np.ndarray, year: int, month: int) -> pd.DataFrame:
    """Create a DataFrame with ride_id and predicted_duration."""
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })
    return df_result

def save_results(df_result: pd.DataFrame, output_file: str) -> None:
    """Save the result DataFrame to a Parquet file."""
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

def main(year: int, month: int):
    """Main function to process data, make predictions, and save results."""
    # Define constants
    categorical = ['PULocationID', 'DOLocationID']
    model_path = 'model.bin'
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'./output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    
    # Ensure output directory exists
    os.makedirs('./output', exist_ok=True)
    
    # Load model
    print(f'Loading the model from {model_path}...')
    dv, model = load_model(model_path)
    
    # Read and preprocess data
    print(f'Reading and processing data from {input_file}...')
    df = read_data(input_file, categorical)
    
    # Make predictions
    print('Making predictions...')
    y_pred = predict(df, dv, model, categorical)
    
    # Print mean of predictions (Q5)
    print(f"The mean of the predictions is: {y_pred.mean():.2f}")
    
    # Create result DataFrame with ride_id
    df_result = create_result_dataframe(df, y_pred, year, month)
    
    # Save results to Parquet
    print(f'Saving the results in {output_file}...')
    save_results(df_result, output_file)

    print('Process Finished Successfully!')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict taxi trip durations and save results.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to process')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to process')
    args = parser.parse_args()
    
    main(args.year, args.month)