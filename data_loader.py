#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:njq
# datetime:2024/11/23 21:53
# software: PyCharm
"""
数据加载和处理
"""
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

def load_data(train_file_path, test_file_path, label_col='Label'):
    """
    加载训练和测试数据并进行预处理。

    参数:
    train_file_path (str): 训练数据文件路径。
    test_file_path (str): 测试数据文件路径。
    label_col (str): 标签列名。

    返回:
    tuple: (X_train, X_test, y_train, y_test)
    """
    train_dataset = pd.read_csv(train_file_path)
    test_dataset = pd.read_csv(test_file_path)

    # 将train_dataset的Label列移动到最后一列
    train_cols = list(train_dataset.columns)
    train_cols.remove(label_col)
    train_cols.append(label_col)
    train_dataset = train_dataset[train_cols]

    test_cols = list(test_dataset.columns)
    test_cols.remove(label_col)
    test_cols.append(label_col)
    test_dataset = test_dataset[test_cols]

    # 删除前面19列
    train_dataset = train_dataset.iloc[:, 17:]
    test_dataset = test_dataset.iloc[:, 17:]

    print(train_dataset.columns)  # 打印最终的列
    
    # 对数据进行打乱，并设置随机种子
    train_data = train_dataset.sample(frac=1, random_state=2025).reset_index(drop=True)
    test_data = test_dataset  # 测试集不需要打乱，方便以后分析使用

    # 分离特征和标签
    X_train = train_data.drop(columns=[label_col])
    y_train = train_data[label_col]
    X_test = test_data.drop(columns=[label_col])
    y_test = test_data[label_col]

    # 特征缺失值填充
    imputer = SimpleImputer(strategy='mean')  # 使用均值填充
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)  # 测试集也使用训练集的参数进行填充

    # 归一化
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)  # 测试集也使用训练集的参数进行填充
    
    # 训练脚本里，fit 之后立刻保存
    dump(imputer, './result/mean_imputer.joblib')
    dump(scaler, './result/minmax_scaler.joblib')

    print("数据加载和预处理完成")
    return X_train, X_test, y_train, y_test

