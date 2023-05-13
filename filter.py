import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


def forward_delete_corr(data):
    # 计算相关系数矩阵
    corr = data.corr().abs()
    # 选取相关系数矩阵的上三角部分
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    # 找出相关系数大于0.7的变量并添加到待删除列表中
    to_delete = [column for column in upper.columns if any(upper[column] > 0.7)]
    print("相关性删除列: ", to_delete)
    return to_delete


def get_low_vif_cols(data, save_path):
    to_delete = []
    # 循环剔除VIF值大于10的变量，直至所有变量的VIF值均小于10
    while True:
        vif = pd.DataFrame()
        vif["variables"] = data.columns
        vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        vif.to_csv(save_path)
        if vif["VIF"].max() > 10:
            # 找出VIF值最大的变量并删除
            col_to_drop = vif.loc[vif["VIF"].idxmax(), "variables"]
            to_delete.append(col_to_drop)
            data = data.drop(col_to_drop, axis=1)
        else:
            break
    print("多重共线性删除列: ", to_delete)
    return to_delete


def get_low_var_cols(data):
    var = data.var()
    to_delete = var[var < 1].index.tolist()
    print("方差删除列: ", to_delete)
    return to_delete


def get_single_enum_cols(data):
    to_delete = []
    for col in data.columns:
        if len(data[col].value_counts()) > 1:
            value_counts = data[col].value_counts(normalize=True)
            if (value_counts >= 0.9).sum() > 0:
                to_delete.append(col)
    print("枚举值删除列: ", to_delete)
    return to_delete
