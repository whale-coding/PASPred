"""
对测试子集做推理，无需重新训练
"""
import pandas as pd
import joblib


MODEL_PATH = './result/model.joblib'  # 训练好的模型

# 加载模型
model = joblib.load(MODEL_PATH)

# 加载训练阶段保存的变换器
imputer = joblib.load('./result/mean_imputer.joblib')
scaler  = joblib.load('./result/minmax_scaler.joblib')

# 读取测试子集
mini_test = pd.read_csv('/data1/niejianqiang/Workspace/9.PASPred/test_mini_score.csv')
# 读取完整测试集
test = pd.read_csv('/data1/niejianqiang/Workspace/9.PASPred/data/3.data_feature/test_feature.csv')

# 匹配出测试子集对应的特征和标签
feat_label = test.merge(mini_test[['pas_id', 'mut_id_hg38']],  # 只拿钥匙列
                  on=['pas_id', 'mut_id_hg38'],
                  how='inner')  # 内连接：只保留突变列表里出现的

# 缺失值填充 + 归一化（用训练集参数）
X_sub = imputer.transform(X_sub)  # mean 填充
X_sub = scaler.transform(X_sub)       # 归一化

# 只保留特征选择后留下来的列以及标签列
wanted = ['CADD_Score','Fathmm_XF_Score','EncodetotalRNA-max','PhyloP','DP_AL',
          'mRNA_expression','RNAplfold_MeanDiff','GerpN','priPhCons','RNAsnp_pValue',
          'EncodeH3K79me2-sum','3mer_AAC','Fathmm-MKL_Score','3mer_TTC','EncodeH3K27me3-max',
          'bStatistic','DP_DL','Roulette-MR','EncodeH3K36me3-sum','GC','DP_DG','DP_AG',
          '2mer_AA','3mer_TGG','Sngl1000bp','EncodeH3K9me3-sum','2mer_AC','2mer_CT',
          '3mer_GGA','EncodeDNase-sum','3mer_GGC','EncodeH2AFZ-sum','3mer_TTA',
          'EncodeH3K4me1-sum','GerpS','3mer_GCA','3mer_GAT','2mer_AG','3mer_CTT',
          '3mer_CAG','EncodeH3K27ac-max','EncodeH3K4me2-sum','EncodeH4K20me1-max',
          '2mer_GA','3mer_TGT','2mer_TT','DS_AG','2mer_CC','CpG','3mer_GGG','Label']

feat_label = feat_label[wanted]
# print(feat_label.shape)
# print(feat_label)

X_sub = feat_label.drop(columns=['Label'])
y_sub = feat_label['Label'].values


# 推理
y_pred = model.predict(X_sub)  # 预测的标签
y_prob = model.predict_proba(X_sub)[:, 1]

# 保存推理结果
results_df = pd.DataFrame({
    'True_Label': y_sub,
    'Predict_Label': y_pred,
    'Probability': y_prob
})

results_df.to_csv(f"./result/model_mini_pred.txt", sep="\t", index=False)

