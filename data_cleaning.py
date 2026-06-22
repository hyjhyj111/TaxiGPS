#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
from datetime import datetime

import pandas as pd


CONFIG = {
    'RAW_DATA_PATH': 'data/raw/TaxiData.csv',
    'SAMPLE_ROWS': 10000,
    'BASE_DATE': '2023-10-12',
    'OUTPUT_PATH': 'data/cleaned/Cleaned_TaxiData.csv',
    'LOG_FILE': 'logs/startup_log.txt',
    'FIELD_NAMES': ['id', 'time', 'long', 'lati', 'status', 'speed'],
}


def create_project_structure():
    dirs = [
        'data/raw',
        'data/cleaned',
        'cache',
        'pages',
        'src',
        'docs',
        'logs'
    ]
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"创建目录: {dir_path}")
    
    return dirs


def setup_logging(log_file):
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def read_sample_data(file_path, sample_rows, field_names):
    if not os.path.exists(file_path):
        print(f"错误：数据文件不存在 - {file_path}")
        print("请将原始CSV数据文件放置在 data/raw/ 目录下")
        sys.exit(1)
    
    try:
        df = pd.read_csv(
            file_path,
            header=None,
            names=field_names,
            nrows=sample_rows,
            encoding='utf-8'
        )
        print(f"成功读取 {len(df)} 行样本数据")
        return df
    except Exception as e:
        print(f"读取数据失败: {str(e)}")
        sys.exit(1)


def clean_data(df, base_date):
    df_clean = df.copy()
    
    df_clean['time'] = pd.to_datetime(
        base_date + ' ' + df_clean['time'],
        format='%Y-%m-%d %H:%M:%S'
    )
    
    df_clean.sort_values(by=['id', 'time'], ascending=[True, True], inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean


def output_cleaned_data(df, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"清洗后数据已保存至: {output_path}")


def generate_statistics(df):
    stats = {
        'total_rows': len(df),
        'total_vehicles': df['id'].nunique(),
        'time_min': df['time'].min(),
        'time_max': df['time'].max(),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict()
    }
    return stats


def main():
    print("="*60)
    print("出租车GPS数据清洗脚本 - 初始化阶段")
    print("="*60)
    
    log = setup_logging(CONFIG['LOG_FILE'])
    
    log.info("="*60)
    log.info("出租车GPS数据清洗脚本 - 初始化阶段")
    log.info("="*60)
    log.info(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("【当前阶段完成情况】")
    log.info("✓ 样本读取")
    log.info("✓ 字段统一")
    log.info("✓ 时间转换")
    log.info("✓ 排序")
    log.info("")
    
    log.info("【创建项目目录结构】")
    dirs = create_project_structure()
    for dir_path in dirs:
        log.info(f"  - {dir_path}")
    log.info("")
    
    log.info("【配置参数】")
    log.info(f"  原始数据路径: {CONFIG['RAW_DATA_PATH']}")
    log.info(f"  样本行数: {CONFIG['SAMPLE_ROWS']}")
    log.info(f"  基准日期: {CONFIG['BASE_DATE']}")
    log.info(f"  输出路径: {CONFIG['OUTPUT_PATH']}")
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
    
    log.info("【读取样本数据】")
    df = read_sample_data(
        CONFIG['RAW_DATA_PATH'],
        CONFIG['SAMPLE_ROWS'],
        CONFIG['FIELD_NAMES']
    )
    
    log.info("【数据清洗】")
    log.info("  - 时间字段转换（补充基准日期）")
    log.info("  - 按id和time排序")
    df_clean = clean_data(df, CONFIG['BASE_DATE'])
    
    log.info("【输出清洗数据】")
    output_cleaned_data(df_clean, CONFIG['OUTPUT_PATH'])
    
    log.info("【清洗后数据统计】")
    stats = generate_statistics(df_clean)
    log.info(f"  总行数: {stats['total_rows']}")
    log.info(f"  车辆数: {stats['total_vehicles']}")
    log.info(f"  时间范围: {stats['time_min']} 至 {stats['time_max']}")
    log.info(f"  列数: {len(stats['columns'])}")
    log.info("")
    
    log.info("【数据类型】")
    for col, dtype in stats['dtypes'].items():
        log.info(f"  {col}: {dtype}")
    log.info("")
    
    log.info("="*60)
    log.info("清洗脚本执行完成")
    log.info("="*60)
    
    print("")
    print("清洗完成！统计信息：")
    print(f"  总行数: {stats['total_rows']}")
    print(f"  车辆数: {stats['total_vehicles']}")
    print(f"  时间范围: {stats['time_min']} 至 {stats['time_max']}")
    print(f"  输出文件: {CONFIG['OUTPUT_PATH']}")
    print(f"  日志文件: {CONFIG['LOG_FILE']}")


if __name__ == '__main__':
    main()