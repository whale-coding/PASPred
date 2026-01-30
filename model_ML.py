#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:njq
# datetime:2025/11/28 19:10
# software: PyCharm
# 训练集和测试集的路径
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import load, dump
from calculate import calculate_metrics, format_metrics
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


train_file_path = "/data1/niejianqiang/Workspace/9.PASPred/data/4.feature_selected/train_selected.csv"
test_file_path = "/data1/niejianqiang/Workspace/9.PASPred/data/4.feature_selected/test_selected.csv"

# 数据加载
train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)

X_train = train.drop(columns=['Label'])
y_train = train['Label']
X_test = test.drop(columns=['Label'])
y_test = test['Label']

# -------------------------------------- K-Nearest Neighbors (KNN) ---------------------------------------
param_dist = {
    'n_neighbors': randint(3, 51),        # k 值范围（奇数可避免平票，但非必须）
    'weights': ['uniform', 'distance'],  # 'distance' 给近邻更高权重，常更优
    'metric': ['euclidean', 'manhattan', 'minkowski'],  # 距离度量
    'p': [1, 2]                          # p=1 → manhattan, p=2 → euclidean（仅当 metric='minkowski' 时有效）
}
model = KNeighborsClassifier()

# -------------------------------------- K-Nearest Neighbors (KNN) ---------------------------------------


# -------------------------------------- Support Vector Machine (SVM) -------------------------------------
# param_dist = {
#     'C': [10**x for x in uniform(-3, 6).rvs(100)],          # 正则强度（越小正则越强）
#     'gamma': [10**x for x in uniform(-4, 5).rvs(100)],      # 核函数系数（'scale'/'auto' 也可考虑）
#     'kernel': ['rbf', 'linear', 'poly'], # 常用核函数
#     'class_weight': ['balanced']         # ⭐ 关键：自动处理不平衡
# }

# model = SVC(
#     probability=True,   # 必须设为 True 才能使用 predict_proba()
#     random_state=42
# )
# -------------------------------------- Support Vector Machine (SVM) -------------------------------------

# -------------------------------------- Naive Bayes (GaussianNB) -----------------------------------------
# param_dist = {
#     'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
# }

# model = GaussianNB()

# -------------------------------------- Naive Bayes (GaussianNB) -----------------------------------------

# -------------------------------------- Logistic Regression (LR) -------------------------------------------
# param_dist = {
#     'C': uniform(0.01, 10),          # 正则强度的倒数：值越小，正则越强
#     'penalty': ['l1', 'l2', 'elasticnet'],  # 正则类型
#     'solver': ['saga'],              # ⭐ 必须用 'saga' 才支持 l1、l2 和 elasticnet
#     'l1_ratio': uniform(0, 1)        # 仅在 penalty='elasticnet' 时生效
# }
# model = LogisticRegression(
#     class_weight='balanced',   # ⭐ 关键：自动调整类别权重
#     random_state=42,
#     max_iter=5000,             # 防止不收敛警告
#     n_jobs=-1
# )

# -------------------------------------- Logistic Regression (LR) -------------------------------------------


# -------------------------------------- Decision Tree (DT) --------------------------------------------------
# param_dist = {
#     'max_depth': randint(3, 20),           # 限制深度防过拟合
#     'min_samples_split': randint(2, 21),   # 内部节点分裂所需最小样本
#     'min_samples_leaf': randint(1, 11),    # 叶子节点最少样本数
#     'max_features': ['sqrt', 'log2', None],# 特征子集选择
#     'criterion': ['gini', 'entropy']       # 分裂标准
# }
# model = DecisionTreeClassifier(
#     class_weight='balanced',   # ⭐ 自动根据类别频率调整权重
#     random_state=42
# )
# -------------------------------------- Decision Tree (DT) --------------------------------------------------

# -------------------------------------- LightGBM --------------------------------------------------
# param_dist = {
#     'n_estimators': randint(100, 600),
#     'max_depth': randint(3, 12),           # LightGBM 默认 leaf-wise，深度可稍大
#     'num_leaves': randint(20, 100),        # 控制模型复杂度（重要！）
#     'learning_rate': uniform(0.01, 0.2),   # [0.01, 0.21)
#     'subsample': uniform(0.6, 0.4),        # 行采样
#     'colsample_bytree': uniform(0.6, 0.4), # 特征采样
#     'min_child_samples': randint(10, 50),  # 叶子节点最少样本数（防过拟合）
#     'reg_alpha': uniform(0, 2),            # L1 正则
#     'reg_lambda': uniform(0, 2),           # L2 正则
#     'scale_pos_weight': [1, 2, 5, 10]      # ⭐ 关键：处理类别不平衡
# }

# model = LGBMClassifier(
#     objective='binary',                    # 二分类
#     metric='auc',                          # 内部评估用 AUC
#     random_state=42,
#     n_jobs=-1,
#     verbosity=-1                           # 减少训练日志输出（可选）
# )
# -------------------------------------- LightGBM --------------------------------------------------

# -------------------------------------- XGBoost --------------------------------------------------
# param_dist = {
#     'n_estimators': randint(100, 500),
#     'max_depth': randint(3, 12),          # XGBoost 对深度更敏感，通常不宜过大
#     'learning_rate': uniform(0.01, 0.2),  # [0.01, 0.21)
#     'subsample': uniform(0.6, 0.4),       # [0.6, 1.0)
#     'colsample_bytree': uniform(0.6, 0.4),# 随机特征采样
#     'gamma': uniform(0, 5),               # 最小损失减少阈值，控制剪枝
#     'min_child_weight': randint(1, 10),   # 控制过拟合，对不平衡数据很重要
#     'scale_pos_weight': [1, 2, 5, 10]     # ⭐ 关键！用于处理不平衡（正样本权重）
# }

# model = XGBClassifier(
#     objective='binary:logistic',
#     eval_metric='auc',        # 内部训练时用 AUC 监控（可选）
#     use_label_encoder=False,  # 消除警告（新版 XGBoost 推荐）
#     random_state=42,
#     n_jobs=-1
# )
# -------------------------------------- XGBoost --------------------------------------------------


# -------------------------------------- RF --------------------------------------------------

# param_dist = {
#     'n_estimators': randint(100, 400),
#     'max_depth': randint(5, 20),
#     'min_samples_split': randint(2, 11),
#     'min_samples_leaf': randint(1, 6),
#     'max_features': ['sqrt', 'log2', None]
# }

# model = RandomForestClassifier(random_state=42)

# -------------------------------------- RF --------------------------------------------------

# ----------------------------------------------------------------------------------------
search = RandomizedSearchCV(
    model, 
    param_distributions=param_dist,
    n_iter=100,          # 尝试100组随机组合
    cv=5,               # 5折交叉验证
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
search.fit(X_train, y_train)

print("Best parameters found:")
print(search.best_params_)
print("Best CV AUC score:", search.best_score_)
# ----------------------------------------------------------------------------------------

# model = RandomForestClassifier(
#     n_estimators=200,          # 树的数量，通常100~500之间
#     max_depth=10,              # 限制树的最大深度，防止过拟合
#     min_samples_split=5,       # 内部节点再划分所需最小样本数
#     min_samples_leaf=2,        # 叶子节点最少样本数
#     max_features='sqrt',       # 每次分裂考虑的特征数（sqrt(n_features) 是常用选择）
#     bootstrap=True,            # 使用自助采样
#     oob_score=True,            # 使用袋外样本来评估泛化精度
#     random_state=42,
#     n_jobs=-1                  # 使用所有CPU核心加速训练
# )
# model.fit(X_train, y_train)

# 模型保存
dump(search.best_estimator_, './result/other/model_KNN.joblib')

# 模型测试
y_pred = search.predict(X_test)
y_proba = search.predict_proba(X_test)[:, 1]  # 预测的概率
# 创建一个 DataFrame 来保存结果
results_df = pd.DataFrame({
    'True_Label': y_test,
    'Predict_Label': y_pred,
    'Probability': y_proba
})
# 保存结果到 CSV 文件
results_df.to_csv(f"./result/other/model_pred_KNN.txt", sep="\t", index=False)


Recall, Specificity, Precision, F1, MCC, Acc, AUC, AUPR = calculate_metrics(y_pred, y_proba, y_test)
print(format_metrics(Recall, Specificity, Precision, F1, MCC, Acc, AUC, AUPR))
# 将评价指标添加到 DataFrame 中
metrics_df = pd.DataFrame({
    'Recall': [Recall],
    'Specificity': [Specificity],
    'Precision': [Precision],
    'F1': [F1],
    'MCC': [MCC],
    'Accuracy': [Acc],
    'AUC': [AUC],
    'AUPR': [AUPR]
})
metrics_df.to_csv(f"./result/other/metrics_KNN.txt", sep="\t", index=False, float_format="%.3f")


