import pandas as pd
import os
import numpy as np
import random
from pydub import AudioSegment
import math
import joblib
from sklearn.decomposition import PCA
""" 从原始CSV文件中提取面部和姿态特征，
并进行时间序列上的标准化（采样、填充/截断），
生成可用于模型训练的特征文件（通常为.npy）。 """                  

def validFrame(frames):
    
    for row in range(frames.shape[0]):
        if frames[row][4] == 1:
            validFrame = frames[row]
    for row in range(frames.shape[0]):
        if frames[row][4] == 0:
            frames[row][5:] = validFrame[5:]
        if frames[row][4] == 1:
            validFrame = frames[row]
    return frames

def chouzhen(_feature):
    flag = 0
    for i in range(0, len(_feature), 6):
        if flag == 0:
            feature = _feature[i]
            flag = 1
        else:
            feature = np.vstack((feature, _feature[i]))
    return feature

def split(data):
    

    _data = chouzhen(data[:5490, ])
    
    if _data.shape[0]<915:
        zeros = np.zeros([(915-_data.shape[0]),_data.shape[1]])
        _data = np.vstack((_data,zeros))
    return _data        

def getTCNVideoFeature(trainPath, targetPath):
    file_list = os.listdir(trainPath)
    
    for file in file_list:
        if file.split(".")[-1] == "csv": 
            full_csv_path = os.path.join(trainPath, file)
            
            try:
                # 1. 读取 CSV 文件，使用 low_memory=False 
                file_csv = pd.read_csv(full_csv_path, low_memory=False)
            except Exception as e:
                print(f'Error reading CSV {file}: {e}')
                continue
            
            data_df = file_csv.copy()
            
            # --- 关键修改 1：清理列名（去除首尾空格），并转换为数字 ---
            data_df.columns = data_df.columns.str.strip()
            
            try:
                # 强制将所有列转换为数字类型，遇到无法转换的值替换为 NaN
                data_df = data_df.apply(pd.to_numeric, errors='coerce')
            except Exception as e:
                print(f'Error converting to numeric DataFrame in {file}: {e}. Skipping.')
                continue
            
            # --- 鲁棒填充 阶段一：时间轴填充 (F-Fill & B-Fill) ---
            
            feature_start_col_index = 5
            
            # 确保 success 列存在，并且索引正确
            try:
                success_col_name = 'success'
                if success_col_name not in data_df.columns:
                    # 假定 success 列依然是第5列 (index 4)
                    success_col_name = data_df.columns[4]
                
                invalid_mask = data_df[success_col_name] == 0
            except Exception as e:
                print(f"Error checking 'success' column in {file}: {e}. Skipping.")
                continue

            feature_cols = data_df.columns[feature_start_col_index:]

            # 1.1 将无效帧的特征数据替换为 NaN 
            data_df.loc[invalid_mask, feature_cols] = np.nan
            
            # 1.2 执行时间轴鲁棒填充 (F-fill, 再 B-fill)
            # 这解决了序列中段的 NaN
            data_df[feature_cols] = data_df[feature_cols].ffill()
            data_df[feature_cols] = data_df[feature_cols].bfill()
            
            # 1.3 🚨 新增：对仍然是 NaN 的帧（即整列/序列都缺失）进行 0 填充
            # 这一步解决了 ffill/bfill 无法解决的头部和尾部缺失问题，也为后续操作提供了干净的数据
            data_df[feature_cols] = data_df[feature_cols].fillna(0)
            
            # ----------------------------------------------------
            
            # 4. 转换为 NumPy 数组
            data = np.array(data_df)
            
            # 5. 执行原有的 split 采样和长度标准化
            try:
                data = split(data) 
            except Exception as e:
                print(f'Video issues in split/chouzhen in {file}: {e}')
                continue
            
            # 6. 删除 OpenFace 头部信息列
            # 0:frame, 1:face_id, 2:timestamp, 3:confidence, 4:success
            data = np.delete(data, [0,1,2,3,4], axis = 1)
            
            # 7. 提取并重组特征 (保持原逻辑不变),代码保持不变，因为它只是 NumPy 切片和拼接) ...
            gaze = data[:, 0:6]
            gaze_zero = np.zeros_like(gaze)
            gaze = np.hstack((gaze, gaze_zero))
            
            pose = data[:, 288:294]
            features = data[:, 294:430]
            au = data[:, 447:465]
            
            au = np.delete(au, [5], axis = 1)
            au[:, [12,13]] = au[:, [13,12]]
            au[:, [13,14]] = au[:, [14,13]] 
            
            feature = au
            feature = np.hstack((feature, features, gaze, pose))
            
            # 8.删除原始的全局均值填充逻辑
            # 由于第 1.3 步已经将所有残余 NaN 填充为 0，这里的逻辑不再需要，并且可以避免 RuntimeWarning
            
            # 9. 最终校验：确保无空值
            if np.isnan(feature).sum() != 0:
                print('CRITICAL ERROR: Residual NaN after all imputation:', file)
                continue 
            
            # 10. 保存 NPY 文件
            np.save(os.path.join(targetPath, file.split(".")[0]), feature)
if __name__ == "__main__":
    
    # 原始CSV文件的父目录
    input_path = "data/LMVD_Feature/Video_feature"
    # 处理后NPY特征的保存目录
    output_path = "data/LMVD_Feature/tcnfeature"
    
    # 检查并创建输出目录
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    getTCNVideoFeature(input_path, output_path)