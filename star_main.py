import filter
import pandas as pd
import preprocess as pp
import model_evaluation as me
import xgboost as xgb
import visualize as vi
import progress as progressCursor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, \
    balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier

if __name__ == '__main__':
    progress = progressCursor.Progress()
    progress.start_progress("开始读取文件")
    # 1.1 读取 star_train.csv 文件
    star_train = pd.read_csv('star_train.csv')
    # 1.2 数据盘点
    progress.show_progress("数据盘点中")
    star_train.describe().to_csv('star_result_data/star_train_describe.csv')
    progress.stop_progress()
    print("盘点完成，请查看 star_train_describe.csv，数据结构可视化如图，关闭窗口后继续")
    # 可视化数据情况
    vi.showFig(star_train)

    # 2.1 缺失值处理-1
    progress.start_progress("缺失值处理中")
    # 计算文件每一列的缺失值比例并保存至 star_train_missing.csv 文件
    star_train.isnull().mean().to_csv('star_result_data/star_train_missing.csv')
    print("缺失比例已保存至 star_train_missing.csv")
    # 去除缺失值比例大于 0.7 的列
    star_train = star_train.loc[:, star_train.isnull().mean() < 0.7]
    progress.stop_progress()
    # 输出处理后的列名
    print("处理完成，留存列名如下:\n", star_train.columns)
    # 输出处理后的数据类型
    print("数据类型如下:\n", star_train.dtypes)

    # 每列根据类型存放在不同列表
    catelist = []
    colist = []
    for i in star_train.columns:
        colist.append(i)
        if star_train[i].dtype == 'object':
            catelist.append(i)
    colist.remove('star_level')
    colist.remove('uid')
    catelist.remove('uid')
    numlist = [i for i in colist if i not in catelist]

    # 2.2 异常值处理
    progress.start_progress("异常值处理中")
    # <2%和>98%的使用2%和98%的数据替换
    star_train = pp.replace_data(star_train, ['star_level'])

    progress.stop_progress()
    print("处理完成，处理后数据结构可视化如图，关闭窗口后继续")
    vi.showFig(star_train)

    # 2.3 数据转换
    progress.start_progress("正在将分类转换为数值类型")
    le = LabelEncoder()
    star_train[catelist] = star_train[catelist].apply(le.fit_transform)
    star_train.to_csv('star_result_data/star_train_trans.csv', index=False)
    print("转化完成，请查看 star_train_trans.csv")

    # 2.4 缺失值处理-2

    # 对于数值型变量，用中位数填充缺失值
    # star_train.loc[:, numlist] = star_train[numlist].fillna(star_train[numlist].median())
    # 对于类别型变量，用众数填充缺失值
    # star_train.fillna(star_train.mode().iloc[0], inplace=True)

    # 检查每一列的数据类型，通过机器学习模型对数值型变量和类别型变量进行填充
    progress.show_progress("使用机器学习处理缺失值中")
    for column in colist:
        # 去除缺失值后获得训练集
        X = star_train[colist].dropna(subset=[column]).drop(columns=[column])
        y = star_train[column].dropna()
        if star_train[column].dtype in ['float64', 'int64']:
            # 使用HistGradientBoostingRegressor类来进行填充数值型变量, 可以自动处理训练集中有缺失特征的情况
            tree_reg = HistGradientBoostingRegressor()
            tree_reg.fit(X, y)
            star_train[column] = star_train[column].fillna(
                pd.Series(tree_reg.predict(star_train[colist].drop(columns=[column])), index=star_train.index))
        elif star_train[column].dtype == 'object':
            # 使用HistGradientBoostingClassifier类来进行填充类别型变量
            tree_clf = HistGradientBoostingClassifier()
            tree_clf.fit(X, y)
            star_train[column] = star_train[column].fillna(
                pd.Series(tree_clf.predict(star_train[colist].drop(columns=[column])), index=star_train.index))

    # 2.5 数据归一化
    # 使用 MinMaxScaler 进行归一化处理
    progress.show_progress("数据归一化中")
    scaler = MinMaxScaler()
    star_train[colist] = scaler.fit_transform(star_train[colist])

    # 2.6 数据标准化
    progress.show_progress("数据标准化中")
    scaler = StandardScaler()
    star_train[colist] = scaler.fit_transform(star_train[colist])
    # 保存处理后的数据至 star_train_clean.csv 文件
    star_train.to_csv('star_result_data/star_train_std.csv', index=False)
    progress.stop_progress()
    print("完成！请查看 star_train_std.csv")

    # 3 特征工程
    progress.start_progress("特征工程处理中")
    # 3.1 计算连续型变量的方差
    # 计算变量方差并保存到 star_train_var.csv 文件
    star_train[colist].var().to_csv('star_result_data/star_train_var.csv')
    # 删除方差小于 1 的变量
    var_to_drop = filter.get_low_var_cols(star_train[numlist])
    colist = [i for i in colist if i not in var_to_drop]

    # 3.2 计算类别型变量枚举值占比
    # 删除枚举值分布集中在单一枚举值上的变量
    fre_to_drop = filter.get_single_enum_cols(star_train[catelist])
    colist = [i for i in colist if i not in fre_to_drop]

    # 3.3 计算变量相关性
    # 计算变量相关性并保存至 star_train_corr.csv 文件
    star_train[colist].corr().to_csv('star_result_data/star_train_corr.csv')
    # 剔除相关性大于 0.7 的变量
    corr_to_drop = filter.forward_delete_corr(star_train[colist])
    colist = [i for i in colist if i not in corr_to_drop]

    # 3.4 计算多重共线性
    # 计算多重共线性并剔除相关性大于 0.7 的变量
    vif_to_drop = filter.get_low_vif_cols(star_train[colist], 'star_result_data/star_train_vif.csv')
    colist = [i for i in colist if i not in vif_to_drop]

    catelist = [i for i in catelist if i in colist]
    numlist = [i for i in numlist if i in colist]
    progress.stop_progress()

    # 4 模型预测
    progress.start_progress("模型训练中")
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(star_train[colist],
                                                        star_train['star_level'],
                                                        test_size=0.3,
                                                        random_state=0)
    # # 4.1 逻辑回归模型
    # # 训练模型
    # lr = LogisticRegression(max_iter=500)
    # lr.fit(X_train, y_train)
    # # 预测
    # y_pred = lr.predict(X_test)

    # # 4.2 决策树模型
    # # 训练模型
    # dt = DecisionTreeClassifier()
    # dt.fit(X_train, y_train)
    # # 预测
    # y_pred = dt.predict(X_test)

    # # 4.3 随机森林模型
    # # 训练模型
    # rf = RandomForestClassifier()
    # rf.fit(X_train, y_train)
    # # 预测
    # y_pred = rf.predict(X_test)

    # 4.4 XGBoost模型
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    # 构建XGBoost分类器
    xgb_clf = xgb.XGBClassifier()
    # 训练XGBoost分类器
    xgb_clf.fit(X_train, y_train)
    # 预测测试集的标签
    y_pred = xgb_clf.predict(X_test)

    progress.stop_progress()

    # 5 模型评估
    # 5.1 计算准确率
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print('模型的准确率为：', accuracy)
    # 5.2 计算平衡准确率为
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print('平衡准确率为:', balanced_acc)
    # 5.3 混淆矩阵
    # 计算混淆矩阵并保存为图片
    # 假设 y_true, y_pred, class_names 已经定义
    save_path = 'star_result_data/confusion_matrix.png'
    class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    me.plot_confusion_matrix(y_test, y_pred, class_names, save_path)
    # 5.4 计算精确率和召回率
    precision = precision_score(y_test, y_pred, average='macro')  # 计算宏平均精确率
    recall = recall_score(y_test, y_pred, average='macro')  # 计算宏平均召回率
    # 打印出来
    print("精确率为: ", precision)
    print("召回率为: ", recall)
    # 5.5 计算F1分数
    f1 = f1_score(y_test, y_pred, average='macro')  # 计算宏平均F1分数
    print("F1分数为: ", f1)
    # 5.6 计算Cohen's Kappa系数
    kappa = cohen_kappa_score(y_test, y_pred)
    print("Cohen's Kappa系数为: ", kappa)

    # 6 模型应用
    star_test = pd.read_csv('star_test.csv')
    # 6.1 数据处理
    # 去重
    star_test = star_test.drop_duplicates()
    # 对于数值型变量，用中位数填充缺失值
    star_test.loc[:, numlist] = star_test[numlist].fillna(star_test[numlist].median())
    # 对于类别型变量，用众数填充缺失值
    star_test.fillna(star_test.mode().iloc[0], inplace=True)
    # 6.1.2 数据转换
    label = LabelEncoder()
    star_test[catelist] = star_test[catelist].apply(label.fit_transform)
    # 6.1.3 数据归一化
    scaler = MinMaxScaler()
    star_test[colist] = scaler.fit_transform(star_test[colist])
    # 6.1.4 数据标准化
    scaler = StandardScaler()
    star_test[colist] = scaler.fit_transform(star_test[colist])
    # 6.2 模型预测
    # 6.2.1 逻辑回归模型
    # y_pred = lr.predict(star_test[colist])
    # 6.2.2 决策树模型
    # y_pred = dt.predict(star_test[colist])
    # 6.2.3 随机森林模型
    # y_pred = rf.predict(star_test[colist])
    # 6.2.4 XGBoost模型
    y_pred = xgb_clf.predict(star_test[colist])
    y_pred = le.inverse_transform(y_pred)
    # 对预测结果进行处理
    y_pred = y_pred.astype(int)
    # 保存预测结果至 star_test_xgb.csv 文件
    star_test['star_level'] = y_pred
    star_test.to_csv('star_result_data/star_test_xgb.csv', index=False)
