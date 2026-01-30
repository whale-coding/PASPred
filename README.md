# PASPred
Prediction tool for human polyadenylation signaling pathogenic mutations based on TabPFN and SHAP。

## 1.Model architecture

![框架图](https://blog-whalecoding.oss-cn-beijing.aliyuncs.com/picgo/20260130110842355.png)

(A) Dataset preparation. Obtain pathogenic single nucleotide mutations from HGMD and Clinvar and obtain benign single nucleotide mutations from Clinvar, and merge them with the human PAS data obtained from the dbHGPS database. According to whether the mutations occur in the PAS region, PAS pathogenic mutations and PAS benign mutations are obtained. Then a data set is constructed and a training set and a test set are divided. (B) Feature encoding. PAS mutations are characterized by coding. Each PAS mutation is characterized by 9 sets of features, namely conservation, signal, splicing, functional score, functional region, mRNA impact, K-mer, genomic background and epigenetic modification features. (C) Feature selection. The FeatureWiz tool is used, combined with the SULOV algorithm and the recursive XGBoost method, to screen the initial feature set. The SULOV algorithm reduces the number of features to 121 dimensions, and then the recursive XGBoost is used to obtain a feature subset containing 50-dimensional features for input to the model. (D) Model construction. Use TabPFN to build a classification model to predict the pathogenicity of PAS mutations. (E) Model explanation. Use SHAP to enhance model interpretability.
