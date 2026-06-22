#!/usr/bin/env python3
import pandas as pd
import os

CONFIG = {
    'CLEANED_DATA_PATH': 'data/cleaned/taxi_cleaned.csv',
    'PICKUP_OUTPUT_PATH': 'data/processed/pickup_points.csv',
    'DROPOFF_OUTPUT_PATH': 'data/processed/dropoff_points.csv'
}

def main():
    print("加载清洗后数据...")
    df = pd.read_csv(CONFIG['CLEANED_DATA_PATH'])
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by=['id', 'time'], inplace=True)
    
    print(f"数据行数: {len(df)}")
    print(f"车辆数: {df['id'].nunique()}")
    
    print("提取上下车点...")
    df['status_chg'] = df.groupby('id')['status'].diff()
    pickup_points = df[df['status_chg'] == 1].copy()
    dropoff_points = df[df['status_chg'] == -1].copy()
    
    print(f"上车点: {len(pickup_points)}")
    print(f"下车点: {len(dropoff_points)}")
    
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    
    pickup_points.to_csv(CONFIG['PICKUP_OUTPUT_PATH'], index=False, encoding='utf-8')
    dropoff_points.to_csv(CONFIG['DROPOFF_OUTPUT_PATH'], index=False, encoding='utf-8')
    print("完成!")

if __name__ == '__main__':
    main()
