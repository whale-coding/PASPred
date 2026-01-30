"""
特征工程
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler

# input_path = "./data/2.final_data/train.csv"
# output_path = "./data/3.data_feature/train_feature.csv"

input_path = "./data/2.final_data/test.csv"
output_path = "./data/3.data_feature/test_feature.csv"

# 1. 先把主表读进来（必须包含 mut_id_hg38）
base = pd.read_csv(input_path)
print('base shape:', base.shape)

# 2. 读取并合并特征
need_cols = ['mut_id_hg38', 'PHRED', 'CADD_Score', 'GC', 'CpG','priPhCons','mamPhCons','verPhCons','priPhyloP','mamPhyloP','verPhyloP','bStatistic',
'GerpN','GerpS','Freq100bp','Rare100bp','Sngl100bp','Freq1000bp','Rare1000bp','Sngl1000bp','Freq10000bp','Rare10000bp','Sngl10000bp','Roulette-MR',
'ZooPriPhyloP','ZooVerPhyloP']
cadd = pd.read_csv('./data/features/CADD_feature.csv', usecols=need_cols).drop_duplicates(subset=['mut_id_hg38'], keep='first') 
base = base.merge(cadd, on='mut_id_hg38', how='left')

# MutationTaster2021
mt = pd.read_csv('./data/features/MutationTaster2021_feature.csv')
mt = mt.drop(columns=['pas_id']).drop_duplicates(subset=['mut_id_hg38'], keep='first') 
base = base.merge(mt, on='mut_id_hg38', how='left')

# RNAsnp
rna = pd.read_csv('./data/features/RNAsnp_feature.csv')
rna = rna.drop(columns=['pas_id']).drop_duplicates(subset=['mut_id_hg38'], keep='first') 
base = base.merge(rna, on='mut_id_hg38', how='left')

# SpliceAI
splice = pd.read_csv('./data/features/SpliceAI_feature.csv')
splice = splice.drop(columns=['pas_id']).drop_duplicates(subset=['mut_id_hg38'], keep='first') 
base = base.merge(splice, on='mut_id_hg38', how='left')

# FATHMM-MKL
fathmm_mk = pd.read_csv('./data/features/FATHMM-MKL_feature.csv')
fathmm_mk = fathmm_mk.drop(columns=['pas_id']).drop_duplicates(subset=['mut_id_hg38'], keep='first') 
base = base.merge(fathmm_mk, on='mut_id_hg38', how='left')

# FATHMM-XF
fathmm_xf = pd.read_csv('./data/features/FATHMM-XF_feature.csv')
fathmm_xf = fathmm_xf.drop(columns=['pas_id']).drop_duplicates(subset=['mut_id_hg38'], keep='first') 
base = base.merge(fathmm_xf, on='mut_id_hg38', how='left')

# mRNA
mrna = pd.read_csv('./data/features/mRNA.csv')
mrna = mrna.drop(columns=['pas_id']).drop_duplicates(subset=['mut_id_hg38'], keep='first') 
base = base.merge(mrna, on='mut_id_hg38', how='left')


# Signal_Pos_preference
spp = pd.read_csv('./data/features/Signal_Pos_preference_feature.csv')
spp = spp.drop(columns=['pas_id']).drop_duplicates(subset=['mut_id_hg38'], keep='first') 
base = base.merge(spp, on='mut_id_hg38', how='left')

# Signal_preference
sp = pd.read_csv('./data/features/Signal_preference_feature.csv')
sp = sp.drop(columns=['pas_id']).drop_duplicates(subset=['mut_id_hg38'], keep='first') 
base = base.merge(sp, on='mut_id_hg38', how='left')

# 3. 保存最终结果
base.to_csv(output_path, index=False)
print('手动合并完成，最终 shape:', base.shape)

