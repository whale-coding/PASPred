#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:njq
# datetime:2024/12/17 16:37
# software: PyCharm
"""
Close by 下采样方式的代码实现
"""
import pandas as pd
from collections import defaultdict


def cb_sampling(positive_samples, negative_samples):
    """Close by 下采样方法，每个负样本最多只被选中一次"""
    balanced_data = positive_samples.copy()

    # 按染色体组织负样本
    negative_by_chrom = defaultdict(list)
    for _, row in negative_samples.iterrows():
        negative_by_chrom[row['chromosome']].append(row)

    # 用于跟踪已选择的负样本
    selected_negatives = set()

    selected_neg_samples = []

    for _, pos_sample in positive_samples.iterrows():
        chrom = pos_sample['chromosome']
        pos = int(pos_sample['Pos'])

        if chrom in negative_by_chrom and negative_by_chrom[chrom]:
            # 找到最近的未被选择的负样本
            available_negs = [neg for neg in negative_by_chrom[chrom]
                              if f"{neg['pas_id']}_{neg['mut_id_hg38']}" not in selected_negatives]

            if available_negs:
                closest_neg = min(available_negs, key=lambda x: abs(int(x['Pos']) - pos))
                # 添加到选中的负样本列表并标记为已选择
                selected_neg_samples.append(closest_neg)
                selected_negatives.add(f"{closest_neg['pas_id']}_{closest_neg['mut_id_hg38']}")

    # 使用concat合并正样本和选中的负样本
    balanced_data = pd.concat([balanced_data, pd.DataFrame(selected_neg_samples)], ignore_index=True)

    return balanced_data
