#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:njq
# datetime:2025/7/5 17:15
# software: PyCharm
"""
评价指标的计算和格式化输出
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, confusion_matrix, matthews_corrcoef


def calculate_metrics(y_pred, y_proba, true_labels):
    """
    计算各种评估指标
    :param y_pred: 模型预测标签
    :param y_proba: 模型预测概率值
    :param true_labels: 真实标签
    :return:
    """
    # Calculate true negatives, false positives, false negatives, and true positives
    tn, fp, fn, tp = confusion_matrix(true_labels, y_pred).ravel()
    # Calculate metrics
    Recall = recall_score(true_labels, y_pred)
    Specificity = tn / (tn + fp)
    MCC = matthews_corrcoef(true_labels, y_pred)
    Precision = precision_score(true_labels, y_pred)
    False_positive_rate = fp / (fp + tn)
    False_negative_rate = fn / (tp + fn)
    F1 = f1_score(true_labels, y_pred)
    Acc = accuracy_score(true_labels, y_pred)
    AUC = roc_auc_score(true_labels, y_proba)
    AUPR = average_precision_score(true_labels, y_proba)
    return Recall, Specificity, Precision, F1, MCC, Acc, AUC, AUPR


# 格式化输出评估指标。
def format_metrics(recall, specificity, precision, f1, mcc, acc, auc, aupr):
    """
    格式化输出评估指标。
    for example:
    test_metrics = calculate_metrics(model_predictions, true_labels, args.threshold)
    print("模型在测试集上的性能：")
    print(format_metrics(*test_metrics))
    """
    metrics_str = (
        f"Recall: {recall:.3f}\n"
        f"Specificity: {specificity:.3f}\n"
        f"Precision: {precision:.3f}\n"
        f"F1: {f1:.3f}\n"
        f"MCC: {mcc:.3f}\n"
        f"Accuracy: {acc:.3f}\n"
        f"AUC: {auc:.3f}\n"
        f"AUPR: {aupr:.3f}"
    )
    return metrics_str
