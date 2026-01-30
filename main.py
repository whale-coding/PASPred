#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:njq
# datetime:2025/7/5 23:24
# software: PyCharm
"""
主函数,这里就包括了特征选择、模型的构建等  conda环境:pytorch
特征选择方法:https://github.com/AutoViML/featurewiz
模型方法:https://github.com/PriorLabs/TabPFN
文章地址:https://www.nature.com/articles/s41586-024-08328-6
"""
import pandas as pd
import numpy as np
import shap
from joblib import load, dump
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

import featurewiz as fw
from tabpfn import TabPFNClassifier

from data_loader import load_data
from test import test_model
import xgboost as xgb
from xgboost import XGBClassifier
from tabpfn_extensions import interpretability
from tabpfn_extensions.interpretability.shap import get_shap_values, plot_shap


# 训练集和测试集的路径
train_file_path = "./data/3.data_feature/train_feature.csv"
test_file_path = "./data/3.data_feature/test_feature.csv"

# 数据加载
X_train, X_test, y_train, y_test = load_data(train_file_path, test_file_path)

# -------------------------------------- 特征选择 -----------------------------------------
# 合并训练集的特征和目标变量
train_combined = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)

target = 'Label'
selected_features, trainm = fw.featurewiz(dataname=train_combined, target=target, corr_limit=0.7, verbose=0)
print('Selected features:', selected_features)
print('featurewiz 选中特征数:', len(selected_features))

X_train_selected = X_train[selected_features] 
X_test_selected = X_test[selected_features]  # 对测试集处理

# 对特征选择后的数据进行保存
train_selected_feature_path = "./data/4.feature_selected/train_selected.csv"
test_selected_feature_path = "./data/4.feature_selected/test_selected.csv"
# 保存训练集（特征 + 标签）
pd.concat([X_train_selected, y_train], axis=1).to_csv(train_selected_feature_path, index=False)
# 保存测试集（特征 + 标签）
pd.concat([X_test_selected, y_test], axis=1).to_csv(test_selected_feature_path, index=False)


# print('\n===== 重新训练并输出特征重要性 =====')
# X = trainm[selected_features]
# y = trainm[target]
# model = XGBClassifier(
#     n_estimators=300,
#     max_depth=6,
#     learning_rate=0.05,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42
# )
# model.fit(X, y)
# importance_df = pd.DataFrame({
#     'feature': selected_features,
#     'importance': model.feature_importances_
# }).sort_values(by='importance', ascending=False)
# imp_path = "./result/xgb_importance_featurewiz.csv"
# importance_df.to_csv(imp_path, index=False)
# print(f'特征重要性已保存至 {imp_path}')
# 排序
# importance_df_sorted = importance_df.sort_values(by="importance", ascending=True)
# plt.figure(figsize=(8, 12))
# plt.barh(
#     importance_df_sorted["feature"],
#     importance_df_sorted["importance"]
# )
# plt.xlabel("Importance")
# plt.title("Feature Importance (XGBoost)")
# plt.tight_layout()
# ⭐ 保存图片（高分辨率）
# plt.savefig("./result/feature_importance.png", dpi=300, bbox_inches='tight')
# plt.show()

# -------------------------------------- 模型构建 -----------------------------------------
# Initialize a classifier
model = TabPFNClassifier()
model.fit(X_train_selected, y_train)

# 模型保存
dump(model, './result/model.joblib')

# -------------------------------------- 模型测试 -----------------------------------------
test_model(model, X_test_selected.values, y_test)


# 特征列改名，方便绘图
# 1. 先建立一个“映射表”：旧名 → 新名
name_map = {
    'CADD_Score': 'CADD score',
    'Fathmm_XF_Score': 'Fathmm-XF score',
    'Fathmm-MKL_Score': 'Fathmm-MKL score',
    # ……按需继续补充
}
# 2. 对两个 DataFrame 同步改名（原地修改）
X_test_selected.rename(columns=name_map, inplace=True)


# -------------------------------------- SHAP分析 -----------------------------------------
feature_names = X_test_selected.columns
print(feature_names)

# Calculate SHAP values
shap_values = get_shap_values(model, X_test_selected)

print(shap_values.values.shape)  # (n_samples, n_features) 或 (n_samples, n_classes, n_features)
print(shap_values.data.shape)    # 应该是 (n_samples, n_features)


plt.figure(figsize=(15, 30))  # 增大图像尺寸以显示所有特征

# 摘要图
shap.summary_plot(
    shap_values.values[:, :, 1],  # 选择第2个类别
    shap_values.data,
    feature_names=feature_names,
    show=False
)


plt.tight_layout()
plt.savefig("./figs/shap摘要图.png", bbox_inches='tight', dpi=300)
plt.close()


# 假设我们关注第2个类别（索引为1）提取该类别的 SHAP 值
shap_values_class = shap.Explanation(
    values=shap_values.values[:, :, 1],
    base_values=shap_values.base_values[:, 1],
    data=shap_values.data,
    feature_names=shap_values.feature_names
)

# 然后再画 bar 图
shap.plots.bar(shap_values_class, show=False, max_display=18, show_data=False)

# 保存Bar Plot图
plt.savefig("./figs/shap重要性图.png", bbox_inches='tight', dpi=300)
plt.close()

# -------------------------------------- SHAP分析单个特征的子图 -----------------------------------------
# 定义绘制 SHAP 依赖图的函数（多个子图）
def plot_shap_dependence(feature_list, df, shap_values_df, file_name="SHAP_Dependence_x1.pdf"):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15), dpi=300)  # 3行3列9个图

    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # 循环绘制每个特征的 SHAP 依赖图
    for i, feature in enumerate(feature_list):
        row = i // 3  # 行号

        col = i % 3  # 列号

        ax = axs[row, col]

        # 绘制散点图，x轴是特征值，y轴是SHAP值
        ax.scatter(df[feature], shap_values_df[feature], s=10)

        # 添加shap=0的横线
        ax.axhline(y=0, color='black', linestyle='-.', linewidth=1)

        # 设置x和y轴标签
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel(f'SHAP value for\n{feature}', fontsize=10)

        # 隐藏顶部和右侧的脊柱
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 设置y轴刻度为2位小数 ← 关键修改
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))

        # 找到SHAP值接近于0的点的x值
        close_to_zero = np.abs(shap_values_df[feature]) < 0.05
        x_values_close_to_zero = df[feature][close_to_zero]
        print(f"Feature: {feature}, x values close to y=0: {x_values_close_to_zero}")
    # 隐藏最后一个空图表的坐标轴 (若画布未关闭)
    if len(feature_list) < 9:
        axs[1, 2].axis('off')

    # 自动调整子图布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(file_name, format='pdf', bbox_inches='tight')


# shap dataframe
shap_values_df = pd.DataFrame(shap_values_class.values, columns=feature_names)

# 使用函数绘制9个特征的shap依赖图
feature_list = ['CADD score', 'Fathmm-XF score', 'priPhCons',
                'EncodetotalRNA-max', '3mer_AAC', 'PhyloP',
                '3mer_TGG', 'Fathmm-MKL score', 'GerpS']

file_name="SHAP_Dependence.pdf"
plot_shap_dependence(feature_list, X_test_selected, shap_values_df,file_name)