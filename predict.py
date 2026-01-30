#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:njq
# datetime:2025/12/29 18:59
# software: PyCharm
"""
使用已经训练好的模型进行预测
"""
import pandas as pd
from joblib import load
from data_loader import load_data

# 训练集和测试集的路径
train_file_path = "./data/3.data_feature/train_feature.csv"
test_file_path = "/data1/niejianqiang/Workspace/9.PASPred/mini/test_mini_fea.csv"

# 数据加载
X_train, X_test, y_train, y_test = load_data(train_file_path, test_file_path)


# # 选择需要的列作为特征
features = ['CADD_Score', 'Fathmm_XF_Score', 'EncodetotalRNA-max', 'PhyloP', 'DP_AL', 'mRNA_expression',
            'RNAplfold_MeanDiff', 'GerpN', 'priPhCons', 'RNAsnp_pValue', 'EncodeH3K79me2-sum', '3mer_AAC',
            'Fathmm-MKL_Score', '3mer_TTC', 'EncodeH3K27me3-max', 'bStatistic', 'DP_DL', 'Roulette-MR',
            'EncodeH3K36me3-sum', 'GC', 'DP_DG', 'DP_AG', '2mer_AA', '3mer_TGG', 'Sngl1000bp', 'EncodeH3K9me3-sum',
            '2mer_AC', '2mer_CT', '3mer_GGA', 'EncodeDNase-sum', '3mer_GGC', 'EncodeH2AFZ-sum', '3mer_TTA',
            'EncodeH3K4me1-sum', 'GerpS', '3mer_GCA', '3mer_GAT', '2mer_AG', '3mer_CTT', '3mer_CAG',
            'EncodeH3K27ac-max', 'EncodeH3K4me2-sum', 'EncodeH4K20me1-max', '2mer_GA', '3mer_TGT', '2mer_TT',
            'DS_AG', '2mer_CC', 'CpG', '3mer_GGG']
X_test = X_test[features]

print("测试集特征形状:", X_test.shape)
print("测试集标签形状:", y_test.shape)

# 读取已经训练好的模型
model = load('./result/model.joblib')

# 进行预测
y_prob = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# 输出预测
print("预测概率:", y_prob)
print("预测结果:", y_pred)

output_df = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred,
    'y_proba': y_prob[:, 1]
})
output_df.to_csv('./predict/prediction_results.csv', index=False)
print("预测结果已保存到 './results/prediction_results.csv'")

from sklearn.metrics import roc_auc_score, average_precision_score
auc = roc_auc_score(y_test, y_prob[:, 1])  # 使用正类的概率计算 AUC
aupr = average_precision_score(y_test, y_prob[:, 1])  # 使用正类的概率计算 AUPR
print(f"AUC: {auc:.4f}")
print(f"AUPR: {aupr:.4f}")

