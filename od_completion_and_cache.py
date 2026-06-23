#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from glob import glob
from tqdm import tqdm


CONFIG = {
    'PICKUP_PATH': 'data/processed/pickup_points.csv',
    'DROPOFF_PATH': 'data/processed/dropoff_points.csv',
    'CLEANED_DATA_PATH': 'data/cleaned/taxi_cleaned.csv',
    'OD_TABLE_PATH': 'data/processed/od_table.csv',
    'VEHICLE_CACHE_DIR': 'cache/vehicles/',
    'MINUTE_CACHE_DIR': 'cache/minutes/',
    'OD_CACHE_PATH': 'cache/od_cache.pkl',
    'MINUTE_RESAMPLE_FREQ': '1min',
    'ABNORMAL_TIME_THRESHOLD': 7200,
    'ABNORMAL_DISTANCE_THRESHOLD': 100,
    'CITY_LAT_MIN': 22.4,
    'CITY_LAT_MAX': 22.6,
    'CITY_LNG_MIN': 113.9,
    'CITY_LNG_MAX': 114.3,
    'LOG_PATH': 'logs/od_cache_startup.log'
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
        CONFIG['VEHICLE_CACHE_DIR'],
        CONFIG['MINUTE_CACHE_DIR'],
        os.path.dirname(CONFIG['OD_CACHE_PATH'])
    ]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"创建目录: {dir_path}")


def haversine_distance(lat1, lng1, lat2, lng2):
    R = 6371.0
    
    lat1_rad = np.radians(lat1)
    lng1_rad = np.radians(lng1)
    lat2_rad = np.radians(lat2)
    lng2_rad = np.radians(lng2)
    
    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlng / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def build_od_table(pickup_df, dropoff_df):
    logging.info("开始 OD 配对")
    
    pickup_df = pickup_df.copy().sort_values(by=['id', 'time']).reset_index(drop=True)
    dropoff_df = dropoff_df.copy().sort_values(by=['id', 'time']).reset_index(drop=True)
    
    pickup_df['event_type'] = 'pickup'
    dropoff_df['event_type'] = 'dropoff'
    
    events = pd.concat([pickup_df, dropoff_df], ignore_index=True)
    events = events.sort_values(by=['id', 'time']).reset_index(drop=True)
    
    orders = []
    current_pickup = None
    
    for _, row in events.iterrows():
        if row['event_type'] == 'pickup':
            if current_pickup is None:
                current_pickup = row
            else:
                current_pickup = row
        else:
            if current_pickup is not None:
                orders.append({
                    'O_TAXI_ID': current_pickup['id'],
                    'O_time': current_pickup['time'],
                    'O_lat': current_pickup['lati'],
                    'O_lng': current_pickup['long'],
                    'O_HEAD': '',
                    'O_SPEED': current_pickup['speed'],
                    'O_FLAG': 1,
                    'D_time': row['time'],
                    'D_lat': row['lati'],
                    'D_lng': row['long'],
                    'D_HEAD': '',
                    'D_SPEED': row['speed'],
                    'D_FLAG': 0
                })
                current_pickup = None
    
    od_df = pd.DataFrame(orders)
    logging.info(f"配对完成，初始订单数: {len(od_df)}")
    
    return od_df


def filter_abnormal_orders(od_df):
    logging.info("开始过滤异常订单")
    
    od_df = od_df.copy()
    
    od_df['O_time'] = pd.to_datetime(od_df['O_time'])
    od_df['D_time'] = pd.to_datetime(od_df['D_time'])
    
    od_df['OD_Time_s'] = (od_df['D_time'] - od_df['O_time']).dt.total_seconds()
    od_df['OD_Dist_km'] = haversine_distance(
        od_df['O_lat'], od_df['O_lng'],
        od_df['D_lat'], od_df['D_lng']
    )
    
    original_count = len(od_df)
    
    time_mask = (od_df['OD_Time_s'] > 0) & (od_df['OD_Time_s'] <= CONFIG['ABNORMAL_TIME_THRESHOLD'])
    distance_mask = (od_df['OD_Dist_km'] > 0) & (od_df['OD_Dist_km'] <= CONFIG['ABNORMAL_DISTANCE_THRESHOLD'])
    
    o_lat_mask = (od_df['O_lat'] >= CONFIG['CITY_LAT_MIN']) & (od_df['O_lat'] <= CONFIG['CITY_LAT_MAX'])
    o_lng_mask = (od_df['O_lng'] >= CONFIG['CITY_LNG_MIN']) & (od_df['O_lng'] <= CONFIG['CITY_LNG_MAX'])
    d_lat_mask = (od_df['D_lat'] >= CONFIG['CITY_LAT_MIN']) & (od_df['D_lat'] <= CONFIG['CITY_LAT_MAX'])
    d_lng_mask = (od_df['D_lng'] >= CONFIG['CITY_LNG_MIN']) & (od_df['D_lng'] <= CONFIG['CITY_LNG_MAX'])
    coord_mask = o_lat_mask & o_lng_mask & d_lat_mask & d_lng_mask
    
    valid_mask = time_mask & distance_mask & coord_mask
    filtered_df = od_df[valid_mask].reset_index(drop=True)
    
    time_abnormal = len(od_df[~time_mask])
    distance_abnormal = len(od_df[~distance_mask])
    coord_abnormal = len(od_df[~coord_mask])
    total_filtered = original_count - len(filtered_df)
    
    logging.info(f"过滤前订单数: {original_count}")
    logging.info(f"时长异常: {time_abnormal}")
    logging.info(f"距离异常: {distance_abnormal}")
    logging.info(f"坐标越界: {coord_abnormal}")
    logging.info(f"过滤后订单数: {len(filtered_df)}")
    
    return filtered_df, {
        'original': original_count,
        'time_abnormal': time_abnormal,
        'distance_abnormal': distance_abnormal,
        'coord_abnormal': coord_abnormal,
        'filtered': len(filtered_df)
    }


def build_vehicle_cache(cleaned_df):
    logging.info("开始构建车辆缓存")
    
    os.makedirs(CONFIG['VEHICLE_CACHE_DIR'], exist_ok=True)
    
    vehicle_groups = cleaned_df.groupby('id')
    vehicle_count = len(vehicle_groups)
    logging.info(f"车辆总数: {vehicle_count}")
    
    for vehicle_id, group in tqdm(vehicle_groups, desc="构建车辆缓存", total=vehicle_count):
        vehicle_id_str = str(int(vehicle_id)) if vehicle_id == int(vehicle_id) else str(vehicle_id)
        cache_file = os.path.join(CONFIG['VEHICLE_CACHE_DIR'], f"{vehicle_id_str}.csv")
        group_sorted = group.sort_values('time').reset_index(drop=True)
        group_sorted.to_csv(cache_file, index=False, encoding='utf-8')
    
    logging.info(f"车辆缓存构建完成，共 {vehicle_count} 个文件")
    return vehicle_count


def build_minute_cache(vehicle_cache_dir):
    logging.info("开始构建分钟缓存")
    
    vehicle_files = glob(os.path.join(vehicle_cache_dir, '*.csv'))
    logging.info(f"待处理车辆文件数: {len(vehicle_files)}")
    
    minute_data = {}
    
    for vehicle_file in tqdm(vehicle_files, desc="处理车辆轨迹"):
        vehicle_id = os.path.basename(vehicle_file).replace('.csv', '')
        try:
            df = pd.read_csv(vehicle_file)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            resampled = df.resample(CONFIG['MINUTE_RESAMPLE_FREQ']).last().ffill()
            resampled['vehicle_id'] = vehicle_id
            
            for idx, row in resampled.iterrows():
                date_str = idx.strftime('%Y-%m-%d')
                hour_str = idx.strftime('%H')
                minute_str = idx.strftime('%M')
                
                key = (date_str, hour_str, minute_str)
                if key not in minute_data:
                    minute_data[key] = []
                
                minute_data[key].append({
                    'vehicle_id': row['vehicle_id'],
                    'long': row['long'],
                    'lati': row['lat'] if 'lat' in row else row['lati'],
                    'status': row['status'],
                    'speed': row['speed']
                })
        except Exception as e:
            logging.warning(f"处理车辆 {vehicle_id} 时出错: {e}")
    
    logging.info(f"生成分钟数据点数: {len(minute_data)}")
    
    for (date_str, hour_str, minute_str), records in tqdm(minute_data.items(), desc="写入分钟缓存"):
        hour_dir = os.path.join(CONFIG['MINUTE_CACHE_DIR'], date_str, hour_str)
        os.makedirs(hour_dir, exist_ok=True)
        
        minute_file = os.path.join(hour_dir, f"{minute_str}.csv")
        df = pd.DataFrame(records)
        df.to_csv(minute_file, index=False, encoding='utf-8')
    
    logging.info("分钟缓存构建完成")
    return len(minute_data)


def build_od_cache(od_df):
    logging.info("开始构建 OD 缓存")
    
    od_cache = {
        'data': od_df,
        'metadata': {
            'created_at': datetime.now(),
            'record_count': len(od_df),
            'columns': od_df.columns.tolist(),
            'time_range': {
                'min': od_df['O_time'].min(),
                'max': od_df['D_time'].max()
            }
        }
    }
    
    with open(CONFIG['OD_CACHE_PATH'], 'wb') as f:
        pickle.dump(od_cache, f)
    
    file_size = os.path.getsize(CONFIG['OD_CACHE_PATH'])
    logging.info(f"OD 缓存构建完成，文件大小: {file_size / 1024 / 1024:.2f} MB")
    return file_size


def verify_outputs():
    logging.info("开始验证输出")
    
    if os.path.exists(CONFIG['OD_TABLE_PATH']):
        od_sample = pd.read_csv(CONFIG['OD_TABLE_PATH'], nrows=5)
        logging.info("OD 表前5条记录:")
        logging.info(od_sample.to_string())
    
    vehicle_files = glob(os.path.join(CONFIG['VEHICLE_CACHE_DIR'], '*.csv'))
    if vehicle_files:
        sample_vehicle = pd.read_csv(vehicle_files[0], nrows=5)
        logging.info(f"\n车辆缓存样本 ({os.path.basename(vehicle_files[0])}):")
        logging.info(sample_vehicle.to_string())
    
    minute_files = glob(os.path.join(CONFIG['MINUTE_CACHE_DIR'], '*', '*', '*.csv'))
    if minute_files:
        sample_minute = pd.read_csv(minute_files[0], nrows=5)
        logging.info(f"\n分钟缓存样本 ({os.path.basename(minute_files[0])}):")
        logging.info(sample_minute.to_string())


def main():
    log = setup_logging(CONFIG['LOG_PATH'])
    
    log.info("="*70)
    log.info("第三阶段：OD 完成与缓存构建")
    log.info("="*70)
    log.info(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("")
    
    log.info("【配置参数】")
    for key, value in CONFIG.items():
        log.info(f"  {key}: {value}")
    log.info("")
    
    create_output_dirs()
    
    log.info("【读取上车点和下车点数据】")
    pickup_df = pd.read_csv(CONFIG['PICKUP_PATH'], encoding='utf-8')
    dropoff_df = pd.read_csv(CONFIG['DROPOFF_PATH'], encoding='utf-8')
    log.info(f"  上车点数量: {len(pickup_df)}")
    log.info(f"  下车点数量: {len(dropoff_df)}")
    log.info("")
    
    log.info("【构建 OD 订单表】")
    od_df = build_od_table(pickup_df, dropoff_df)
    od_df, filter_stats = filter_abnormal_orders(od_df)
    od_df.to_csv(CONFIG['OD_TABLE_PATH'], index=False, encoding='utf-8')
    log.info(f"  OD 表已保存至: {CONFIG['OD_TABLE_PATH']}")
    log.info("")
    
    log.info("【构建车辆缓存】")
    cleaned_df = pd.read_csv(CONFIG['CLEANED_DATA_PATH'], encoding='utf-8')
    cleaned_df['time'] = pd.to_datetime(cleaned_df['time'])
    vehicle_count = build_vehicle_cache(cleaned_df)
    log.info("")
    
    log.info("【构建分钟缓存】")
    minute_count = build_minute_cache(CONFIG['VEHICLE_CACHE_DIR'])
    log.info("")
    
    log.info("【构建 OD 缓存】")
    cache_size = build_od_cache(od_df)
    log.info("")
    
    log.info("【验证输出】")
    verify_outputs()
    log.info("")
    
    log.info("【执行摘要】")
    log.info(f"  原始上车点: {len(pickup_df)}")
    log.info(f"  原始下车点: {len(dropoff_df)}")
    log.info(f"  配对后订单: {filter_stats['original']}")
    log.info(f"  过滤异常订单: {filter_stats['original'] - filter_stats['filtered']}")
    log.info(f"    - 时长异常: {filter_stats['time_abnormal']}")
    log.info(f"    - 距离异常: {filter_stats['distance_abnormal']}")
    log.info(f"    - 坐标越界: {filter_stats['coord_abnormal']}")
    log.info(f"  有效订单数: {filter_stats['filtered']}")
    log.info(f"  车辆缓存文件数: {vehicle_count}")
    log.info(f"  分钟缓存文件数: {minute_count}")
    log.info(f"  OD 缓存大小: {cache_size / 1024 / 1024:.2f} MB")
    log.info("")
    
    log.info("="*70)
    log.info("第三阶段完成")
    log.info("="*70)


if __name__ == '__main__':
    main()