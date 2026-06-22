#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
from datetime import datetime

import pandas as pd


CONFIG = {
    'RAW_DATA_PATH': 'data/raw/TaxiData.csv',
    'CLEANED_DATA_PATH': 'data/cleaned/taxi_cleaned.csv',
    'LOG_PATH': 'logs/clean_od_startup.log',
    'BASE_DATE': '2023-10-12',
    'CHUNK_SIZE': 5000000,
    'TIME_THRESHOLD_SEC': 60,
    'COLUMN_NAMES': ['id', 'time', 'long', 'lati', 'status', 'speed'],
    'PICKUP_OUTPUT_PATH': 'data/processed/pickup_points.csv',
    'DROPOFF_OUTPUT_PATH': 'data/processed/dropoff_points.csv'
}


def setup_logging(log_path):
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def create_output_dirs():
    dirs = [
        'data/raw',
        'data/cleaned',
        'data/processed',
        'logs'
    ]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"创建目录: {dir_path}")


def process_duplicates(df):
    """处理重复记录"""
    duplicates = df.duplicated(subset=['id', 'time'], keep=False)
    if not duplicates.any():
        return df, 0, 0
    
    duplicate_df = df[duplicates].copy()
    original_len = len(df)
    
    def keep_strategy(group):
        if len(group) == 1:
            return group.iloc[[0]]
        
        status_vals = group['status'].values
        unique_status = len(set(status_vals))
        
        if unique_status == 1:
            return group.iloc[[0]]
        else:
            if len(group) == 2:
                if 0 in status_vals and 1 in status_vals:
                    return group[group['status'] == 0].iloc[[0]]
                else:
                    return group.iloc[[0]]
            elif len(group) == 3:
                status_sum = status_vals.sum()
                if status_sum == 1:
                    return group[group['status'] == 0].iloc[[0]]
                elif status_sum == 2:
                    return group[group['status'] == 1].iloc[[0]]
                else:
                    return group.iloc[[0]]
            else:
                return group.iloc[[0]]
    
    cleaned_duplicates = duplicate_df.groupby(['id', 'time'], group_keys=False).apply(keep_strategy)
    non_duplicates = df[~duplicates]
    result = pd.concat([non_duplicates, cleaned_duplicates], ignore_index=True)
    
    removed = original_len - len(result)
    return result, removed


def identify_anomalies(df, time_threshold_sec):
    """识别并剔除异常状态记录"""
    df = df.copy()
    df['prev_status'] = df.groupby('id')['status'].shift(1)
    df['next_status'] = df.groupby('id')['status'].shift(-1)
    df['prev_id'] = df.groupby('id')['id'].shift(1)
    df['next_id'] = df.groupby('id')['id'].shift(-1)
    df['prev_time'] = df.groupby('id')['time'].shift(1)
    df['next_time'] = df.groupby('id')['time'].shift(-1)
    
    df['time_diff_prev'] = (df['time'] - df['prev_time']).dt.total_seconds()
    df['time_diff_next'] = (df['next_time'] - df['time']).dt.total_seconds()
    
    anomaly_mask = (
        (df['status'] != df['prev_status']) &
        (df['status'] != df['next_status']) &
        (df['id'] == df['prev_id']) &
        (df['id'] == df['next_id']) &
        (df['time_diff_prev'] < time_threshold_sec) &
        (df['time_diff_next'] < time_threshold_sec)
    )
    
    anomalies = df[anomaly_mask]
    cleaned_df = df[~anomaly_mask].drop(columns=['prev_status', 'next_status', 'prev_id', 'next_id', 'prev_time', 'next_time', 'time_diff_prev', 'time_diff_next'])
    
    return cleaned_df, len(anomalies)


def clean_chunk(df, base_date, time_threshold_sec):
    """清洗单个数据块"""
    df_clean = df.copy()
    
    df_clean['time'] = pd.to_datetime(
        base_date + ' ' + df_clean['time'],
        format='%Y-%m-%d %H:%M:%S'
    )
    
    df_clean.sort_values(by=['id', 'time'], ascending=[True, True], inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    
    df_clean, dup_removed = process_duplicates(df_clean)
    
    df_clean, anomaly_removed = identify_anomalies(df_clean, time_threshold_sec)
    
    return df_clean, dup_removed, anomaly_removed


def process_with_boundary(chunk_iter, base_date, time_threshold_sec):
    """带边界处理的分块清洗流程"""
    total_original = 0
    total_cleaned = 0
    total_dup_removed = 0
    total_anomaly_removed = 0
    prev_chunk_last_row = None
    first_chunk = True
    
    if os.path.exists(CONFIG['CLEANED_DATA_PATH']):
        os.remove(CONFIG['CLEANED_DATA_PATH'])
    
    for chunk_idx, chunk in enumerate(chunk_iter):
        chunk.columns = CONFIG['COLUMN_NAMES']
        original_len = len(chunk)
        total_original += original_len
        
        if prev_chunk_last_row is not None:
            chunk = pd.concat([prev_chunk_last_row, chunk], ignore_index=True)
        
        logging.info(f"处理分块 {chunk_idx + 1}，原始行数: {len(chunk)}")
        
        cleaned_chunk, dup_removed, anomaly_removed = clean_chunk(chunk, base_date, time_threshold_sec)
        
        if len(cleaned_chunk) > 0:
            prev_chunk_last_row = cleaned_chunk.iloc[[-1]].copy()
            cleaned_chunk = cleaned_chunk.iloc[:-1]
        
        cleaned_chunk.to_csv(
            CONFIG['CLEANED_DATA_PATH'],
            mode='a',
            header=first_chunk,
            index=False,
            encoding='utf-8'
        )
        first_chunk = False
        
        total_cleaned += len(cleaned_chunk)
        total_dup_removed += dup_removed
        total_anomaly_removed += anomaly_removed
        
        logging.info(f"  分块 {chunk_idx + 1} 清洗完成")
        logging.info(f"    - 清洗后行数: {len(cleaned_chunk)}")
        logging.info(f"    - 剔除重复数: {dup_removed}")
        logging.info(f"    - 剔除异常数: {anomaly_removed}")
    
    if prev_chunk_last_row is not None:
        prev_chunk_last_row.to_csv(
            CONFIG['CLEANED_DATA_PATH'],
            mode='a',
            header=False,
            index=False,
            encoding='utf-8'
        )
        total_cleaned += len(prev_chunk_last_row)
    
    return total_original, total_cleaned, total_dup_removed, total_anomaly_removed


def extract_od_points(df):
    """从清洗后数据中提取上下车点"""
    df_sorted = df.sort_values(by=['id', 'time']).copy()
    df_sorted['status_chg'] = df_sorted.groupby('id')['status'].diff()
    
    pickup_points = df_sorted[df_sorted['status_chg'] == 1].copy()
    dropoff_points = df_sorted[df_sorted['status_chg'] == -1].copy()
    
    return pickup_points, dropoff_points


def main():
    log = setup_logging(CONFIG['LOG_PATH'])
    
    log.info("="*70)
    log.info("出租车GPS数据清洗与OD提取脚本 - 第二阶段")
    log.info("="*70)
    log.info(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("")
    
    log.info("【配置参数】")
    for key, value in CONFIG.items():
        log.info(f"  {key}: {value}")
    log.info("")
    
    log.info("【字段说明】")
    field_desc = {
        'id': '车辆唯一标识（字符串或整数）',
        'time': 'GPS采集时间（datetime64类型）',
        'long': '经度（浮点数）',
        'lati': '纬度（浮点数）',
        'status': '载客状态（1=载客，0=空客）',
        'speed': '瞬时速度（km/h，浮点数）'
    }
    for field, desc in field_desc.items():
        log.info(f"  {field}: {desc}")
    log.info("")
    
    log.info("【创建输出目录】")
    create_output_dirs()
    log.info("")
    
    if not os.path.exists(CONFIG['RAW_DATA_PATH']):
        log.error(f"错误：原始数据文件不存在 - {CONFIG['RAW_DATA_PATH']}")
        log.error("请将原始CSV数据文件放置在 data/raw/ 目录下")
        sys.exit(1)
    
    log.info("【开始分块读取与清洗】")
    chunk_iter = pd.read_csv(
        CONFIG['RAW_DATA_PATH'],
        header=None,
        chunksize=CONFIG['CHUNK_SIZE'],
        encoding='utf-8'
    )
    
    total_original, total_cleaned, total_dup_removed, total_anomaly_removed = process_with_boundary(
        chunk_iter,
        CONFIG['BASE_DATE'],
        CONFIG['TIME_THRESHOLD_SEC']
    )
    log.info("")
    
    log.info("【加载清洗后数据】")
    df_final = pd.read_csv(CONFIG['CLEANED_DATA_PATH'], encoding='utf-8')
    df_final['time'] = pd.to_datetime(df_final['time'])
    df_final.sort_values(by=['id', 'time'], ascending=[True, True], inplace=True)
    df_final.reset_index(drop=True, inplace=True)
    log.info(f"  加载后总行数: {len(df_final)}")
    log.info("")
    
    log.info("【清洗摘要】")
    log.info(f"  原始总行数: {total_original}")
    log.info(f"  清洗后总行数: {len(df_final)}")
    log.info(f"  剔除重复行数: {total_dup_removed}")
    log.info(f"  剔除异常行数: {total_anomaly_removed}")
    log.info(f"  车辆数: {df_final['id'].nunique()}")
    log.info(f"  时间范围: {df_final['time'].min()} 至 {df_final['time'].max()}")
    log.info("")
    
    log.info("【清洗规则说明】")
    log.info("  1. 重复值处理:")
    log.info("     - 同一id+time的记录，若status相同，保留第一条")
    log.info("     - 重复数为2且status不同(0和1)，保留status=0的记录")
    log.info("     - 重复数为3，按多数原则保留")
    log.info("  2. 异常状态剔除:")
    log.info(f"     - 孤立状态(0-1-0或1-0-1)且前后时间差<{CONFIG['TIME_THRESHOLD_SEC']}秒")
    log.info("")
    
    log.info("【开始OD提取】")
    pickup_points, dropoff_points = extract_od_points(df_final)
    
    log.info(f"  上车点总数: {len(pickup_points)}")
    log.info(f"  下车点总数: {len(dropoff_points)}")
    log.info("")
    
    log.info("【上车点样本（前10行）】")
    if len(pickup_points) > 0:
        log.info(pickup_points.head(10).to_string())
    log.info("")
    
    log.info("【下车点样本（前10行）】")
    if len(dropoff_points) > 0:
        log.info(dropoff_points.head(10).to_string())
    log.info("")
    
    log.info("【保存上下车点数据】")
    pickup_points.to_csv(CONFIG['PICKUP_OUTPUT_PATH'], index=False, encoding='utf-8')
    dropoff_points.to_csv(CONFIG['DROPOFF_OUTPUT_PATH'], index=False, encoding='utf-8')
    log.info(f"  上车点已保存至: {CONFIG['PICKUP_OUTPUT_PATH']}")
    log.info(f"  下车点已保存至: {CONFIG['DROPOFF_OUTPUT_PATH']}")
    log.info("")
    
    log.info("="*70)
    log.info("清洗与OD提取脚本执行完成")
    log.info("="*70)
    
    print("\n" + "="*70)
    print("清洗与OD提取完成！")
    print("="*70)
    print(f"原始总行数: {total_original}")
    print(f"清洗后总行数: {len(df_final)}")
    print(f"剔除重复行数: {total_dup_removed}")
    print(f"剔除异常行数: {total_anomaly_removed}")
    print(f"车辆数: {df_final['id'].nunique()}")
    print(f"时间范围: {df_final['time'].min()} 至 {df_final['time'].max()}")
    print(f"上车点总数: {len(pickup_points)}")
    print(f"下车点总数: {len(dropoff_points)}")
    print(f"清洗数据文件: {CONFIG['CLEANED_DATA_PATH']}")
    print(f"日志文件: {CONFIG['LOG_PATH']}")
    print("="*70)


if __name__ == '__main__':
    main()