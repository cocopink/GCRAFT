#!/usr/bin/env python3
"""
从 cache 目录中合并单个 case 的缓存文件，生成 final_result.pkl

用法:
    python merge_cache_to_final.py --cache_dir <cache_dir> --output_dir <output_dir>

示例:
    python merge_cache_to_final.py --cache_dir results/thought_100/tabfact/gpt-5.4/cache --output_dir results/thought_100/tabfact/gpt-5.4
"""

import os
import pickle
import argparse
from tqdm import tqdm
import glob


def read_pkl(pkl_file):
    """读取 pkl 文件"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data


def merge_cache_to_final(cache_dir, output_dir):
    """
    从 cache 目录中合并单个 case 的缓存文件，生成 final_result.pkl
    
    Args:
        cache_dir: cache 目录路径
        output_dir: 输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有 cache 文件，按索引排序
    cache_files = sorted(
        glob.glob(os.path.join(cache_dir, "case-*.pkl")),
        key=lambda x: int(os.path.basename(x).split('-')[1].split('.')[0])
    )
    
    if not cache_files:
        print(f"错误: 在 {cache_dir} 中没有找到 cache 文件")
        return None
    
    print(f"找到 {len(cache_files)} 个 cache 文件")
    
    # 读取所有 cache 文件，提取处理后的样本
    final_result = []
    
    for cache_file in tqdm(cache_files, desc="合并 cache 文件"):
        try:
            # cache 文件格式: (sample, proc_sample, log)
            data = read_pkl(cache_file)
            
            if isinstance(data, tuple) and len(data) >= 2:
                # 提取处理后的样本 (proc_sample)
                proc_sample = data[1]
                final_result.append(proc_sample)
            else:
                print(f"警告: {cache_file} 的数据格式不正确，跳过")
        except Exception as e:
            print(f"错误: 读取 {cache_file} 失败: {e}")
            continue
    
    print(f"成功合并 {len(final_result)} 个样本")
    
    # 保存 final_result.pkl
    output_path = os.path.join(output_dir, "final_result.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(final_result, f)
    
    print(f"已保存到: {output_path}")
    
    return final_result


def main():
    parser = argparse.ArgumentParser(description='从 cache 目录合并单个 case 的缓存文件，生成 final_result.pkl')
    parser.add_argument('--cache_dir', type=str, required=True, help='cache 目录路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录路径')
    
    args = parser.parse_args()
    
    merge_cache_to_final(args.cache_dir, args.output_dir)


if __name__ == "__main__":
    main()
