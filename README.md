# Simple-Predict

根据客户在银行中的各种信息，通过机器学习对其进行星级评估与信用评估

由于数据为银行中的真实数据，存在敏感信息，故仅给出字段

## 1. 数据收集

### 1.1 用户星级

| 属性名         | 属性解释                   | 选择理由                                                     |
| -------------- | -------------------------- | ------------------------------------------------------------ |
| 贷记卡开户明细 | djk_info                   | 如果夫妻双方均有稳定的收入来源，通常意味着他们有更高的存款能力；但婚姻状态通常也伴随着家庭开销的增加 |
| deposit        | 贷记卡存款                 | 存款金额较高，说明用户有一定的储蓄能力和可支配收入           |
| bal            | 余额                       | 贷记卡余额较低，说明用户有较好的储蓄能力和财务规划能力。相反，如果贷记卡余额较高，则可能表明用户缺乏储蓄能力或存在过度消费的风险 |
| bankacct_bal   | 还款账号余额               | 还款账号余额较高，说明用户具有一定的储蓄能力和资金储备，有能力应对紧急支出和应急情况 |
| 基本信息       | pri_cust_base_info         |                                                              |
| marrige        | 婚姻状况                   |                                                              |
| education      | 教育程度                   | 一定程度上可以反映用户的未来存款稳定性和潜力                 |
| career         | 职业                       | 不同职业的收入水平有所不同，收入稳定性也不同                 |
| prof_titl      | 职称                       | 反映用户在同种职业中的相对存款能力                           |
| is_black       | 是否黑名单                 | 黑名单用户在星级评分上得分更低                               |
| is_contact     | 是否关联人                 | 关联人的信用水平对账户持有人的信用水平产生一致的影响         |
| 存款汇总信息   | pri_cust_asset_info        |                                                              |
| all_bal        | 总余额                     | 反映用户的整体存款能力                                       |
| avg_mth        | 月日均，表示⽉平均余额     | 反映用户的月度存款水平                                       |
| avg_qur        | 季度日均，表示季度平均余额 | 反映用户的季度存款水平                                       |
| avg_year       | 年日均，表示年度平均余额   | 反映用户的年度存款水平                                       |
| sa_bal         | 活期余额                   | 通过活期余额的绝对数量和占总余额的比例，推算用户的存款习惯和能力 |
| td_bal         | 定期余额                   | 通过定期余额的绝对数量和占总余额的比例，推算用户的存款习惯和能力 |
| fin_bal        | 理财余额                   | 通过理财余额的绝对数量和占总余额的比例，推算用户的存款习惯和能力 |
| sa_crd_bal     | 卡活期余额                 | 通过卡活期余额的绝对数量和占总余额的比例，推算用户的存款习惯和能力 |
| td_crd_bal     | 卡内定期                   | 通过卡内定期余额的绝对数量和占总余额的比例，推算用户的存款习惯和能力 |
| sa_td_bal      | 定活两便                   | 通过定活两便余额的绝对数量和占总余额的比例，推算用户的存款习惯和能力 |
| ntc_bal        | 通知存款                   | 通过通知存款的绝对数量和占总余额的比例，推算用户的存款习惯和能力 |
| td_1y_bal      | 定期1年                    | 用户定期存款的一部分                                         |
| td_2y_bal      | 定期2年                    | 用户定期存款的一部分                                         |
| td_3y_bal      | 定期3年                    | 用户定期存款的一部分                                         |
| td_5y_bal      | 定期5年                    | 用户定期存款的一部分                                         |
| td_3m_bal      | 定期3个⽉                  | 用户定期存款的一部分                                         |
| td_6m_bal      | 定期6个⽉                  | 用户定期存款的一部分                                         |
| oth_td_bal     | 定期其他余额               | 通过定期其他余额的绝对数量和占总余额的比例，推算用户的存款习惯和能力 |
| cd_bal         | 大额存单余额               | 通过大额存单余额的绝对数量和占总余额的比例，推算用户的存款习惯和能力 |
| 存款账号信息   | pri_cust_asset_acct_info   |                                                              |
| acct_sts       | 账户状态                   | 健康的账户状态对于用户星级评分是正向的                       |
| frz_sts        | 冻结状态                   | 异常的冻结状态对于用户星级评分是负向的                       |
| stp_sts        | 止付状态                   | 止付状态本身并不能反映用户的存款能力，但其背后可能反映出用户的一些财务问题和风险 |
| acct_bal       | 账户余额                   | 直接反映用户存款水平                                         |
| bal            | 余额                       | 反映用户存款水平                                             |

### 1.2 信用等级

| 属性名         | 属性解释             | 选择理由                                                     |
| -------------- | -------------------- | ------------------------------------------------------------ |
| 基本信息       | pri_cust_base_info   |                                                              |
| marrige        | 婚姻状况             | 如果夫妻双方均有稳定的收入来源，通常意味着他们有更高的存款能力；但婚姻状态通常也伴随着家庭开销的增加 |
| education      | 教育程度             | 一定程度上可以反映用户的未来存款稳定性和潜力                 |
| career         | 职业                 | 不同职业的收入水平有所不同，收入稳定性也不同                 |
| prof_titl      | 职称                 | 反映用户当前阶段的存款能力                                   |
| is_black       | 是否黑名单           | 黑名单用户在星级评分上得分更低                               |
| is_contact     | 是否关联人           | 关联人的信用水平对账户持有人的信用水平产生一致的影响         |
| 贷记卡开户明细 | djk_info             |                                                              |
| cred_limit     | 信用额度             | 对用户信用水平的预估，信用额度越大往往信用等级越高           |
| over_draft     | 普通额度透支         | 对用户信用水平的预估，透支额度越大往往信用等级越高           |
| dlay_amt       | 逾期金额             | 用户信用行为的表现，逾期金额越大往往意味着更差的信用         |
| 合同明细       | dm_v_tr_contract_mx  |                                                              |
| dull_bal       | 呆滞余额（超时90天） | 呆滞余额越高，说明用户信用越差                               |
| owed_int_in    | 表内欠息金额         | 表内欠息金额越多，说明用户信用越差                           |
| owed_int_out   | 表外欠息金额         | 表外欠息金额越多，说明用户信用越差                           |
| extend_times   | 展期次数             | 展期次数越多，说明用户信用越差                               |
| vouch_type     | 主要担保方式         | 以不动产为基准对担保方式进行评分，有更高可靠性的担保方式的用户，信用越高 |
| fine_pr_int    | 本金罚息             | 本金罚息越多，说明用户信用越差                               |
| fine_intr_int  | 利息罚息             | 利息罚息越多，说明用户信用越差                               |
| five_class     | 五级分类             | 贷款的五级分类，分类等级越高，贷款性质越差，对应的用户信用越低 |
| dlay_bal       | 逾期余额             | 逾期余额越多，说明用户信用越差                               |
| 借据明细       | dm_v_tr_duebill_mx   |                                                              |
| owed_int_in    | 表内欠息金额         | 表内欠息金额越多，说明用户信用越差                           |
| owed_int_out   | 表外欠息金额         | 表外欠息金额越多，说明用户信用越差                           |
| extend_times   | 展期次数             | 展期次数越多，说明用户信用越差                               |
| vouch_type     | 主要担保方式         | 以不动产为基准对担保方式进行评分，有更高可靠性的担保方式的用户，信用越高 |
| fine_intr_int  | 利息罚息             | 利息罚息越多，说明用户信用越差                               |
| fine_pr_int    | 本金罚息             | 本金罚息越多，说明用户信用越差                               |
| dlay_days      | 逾期天数             | 逾期天数越多，说明用户信用越差                               |
| dlay_bal       | 逾期余额             | 逾期余额越多，说明用户信用越差                               |
| dull_bal       | 呆滞余额（超时90天） | 呆滞余额越多，说明用户信用越差                               |
| due_intr_days  | 欠息天数             | 欠息天数越多，说明用户信用越差                               |
| ten_class      | 新十级分类编码       | 贷款的十级分类，分类等级越高，贷款性质越差，对应的用户信用越低 |
| 贷款账户汇总   | pri_cust_liab_info   |                                                              |
| all_bal        | 总余额               | 贷款账户的总余额越多，说明用户欠款越多，需要较高的还款能力进行偿还，变为坏账的风险就越高 |
| bad_bal        | 不良余额             | 不良余额是已经逾期并且被认为可能无法收回的未偿还本金和利息，不良余额越大，说明用户信用越差 |
| due_intr       | 欠息总额             | 用户欠息总额越大，用户信用越差                               |
| norm_bal       | 正常余额             | 用户正常余额是指未逾期，并在可收回范围内的本金和利息，如果正常余额在用户的偿还能力之内则没有问题 |
| delay_bal      | 逾期总额             | 已经逾期的未偿还本金和利息的总和，逾期总额越大，说明用户信用越差 |



## 2. 数据盘点

### 2.1 数据信息盘点

本次作业的数据盘点部分基于 `baseline.py`，使用 pandas 提供的数据盘点功能对各列数据的总数、平均数、标准差、极值和 1/4、1/2、3/4 分位数进行计算并保存为表格。例如对于 `star` 选中的数据进行计算的结果如下：

|       | **all_bal**        | **sa_bal**        | **td_bal**         | **fin_bal**        | **sa_crd_bal**    | **td_crd_bal** | **sa_td_bal**      | **ntc_bal**        | **oth_td_bal** | **cd_bal**         | **asset_bal**     | **deposit**        | **bal** | **bankacct_bal**   | **star_level**     |
| ----- | ------------------ | ----------------- | ------------------ | ------------------ | ----------------- | -------------- | ------------------ | ------------------ | -------------- | ------------------ | ----------------- | ------------------ | ------- | ------------------ | ------------------ |
| count | 299136.0           | 299136.0          | 299136.0           | 299136.0           | 299136.0          | 299136.0       | 299136.0           | 299136.0           | 299136.0       | 299136.0           | 194420.0          | 17974.0            | 0.0     | 9476.0             | 299147.0           |
| mean  | 42451.04595234275  | 6355.277522598417 | 35191.35595515084  | 898.9556589644844  | 5957.099423004921 | 0.0            | 1038.9213768988018 | 418.00863486842104 | 0.0            | 5958.961810012837  | 2361.587331550252 | 114.07499721820408 |         | 14098.033452933727 | 1.9632154091466738 |
| std   | 168559.30297591496 | 39030.16825788596 | 155520.72303531636 | 24172.863917450442 | 38360.38003829735 | 0.0            | 17350.55264230346  | 16233.597605313744 | 0.0            | 121668.28202124593 | 23171.02124644373 | 3656.8348079760185 |         | 76733.57789293845  | 1.2548572132058915 |
| min   | 0.0                | 0.0               | 0.0                | 0.0                | 0.0               | 0.0            | 0.0                | 0.0                | 0.0            | 0.0                | 0.0               | 0.0                |         | 0.0                | 1.0                |
| 25%   | 52.0               | 26.0              | 0.0                | 0.0                | 9.0               | 0.0            | 0.0                | 0.0                | 0.0            | 0.0                | 0.0               | 0.0                |         | 10.0               | 1.0                |
| 50%   | 1105.0             | 456.0             | 0.0                | 0.0                | 373.0             | 0.0            | 0.0                | 0.0                | 0.0            | 0.0                | 6.0               | 0.0                |         | 424.0              | 1.0                |
| 75%   | 24784.0            | 2712.0            | 10000.0            | 0.0                | 2416.0            | 0.0            | 0.0                | 0.0                | 0.0            | 0.0                | 374.0             | 0.0                |         | 5223.5             | 3.0                |
| max   | 25320222.0         | 3341648.0         | 24000000.0         | 2000000.0          | 3341648.0         | 0.0            | 2550000.0          | 2002101.0          | 0.0            | 24000000.0         | 3341648.0         | 312508.0           |         | 2650685.0          | 9.0                |

### 2.2 数据可视化展示

我们选择使用 `matplotlib` 包下的 `pyplot` 以及 `pandas` 中集成的 `pyplot` 将数据可视化为频数分布直方图进行展示，能够清晰地观察到数据的分布特点。

我们将可视化步骤抽取为单独的模块，只需传入 `dataframe` 即可进行可视化。在模块内部，我们会根据数据的不同类型选用不同的方法。

对于值为类型的列首先使用 `pandas` 统计各类型的数量，然后使用 `pandas` 中集成的 `plot` 绘制柱状图，并手动为柱状图添加数量显示，代码如下：

```python
# 如果是类别型数据，先统计不同类别数量再直接画柱形图
if df[col].dtype == "object" or col == 'star_level':
  # value_counts 统计数量，sort_index 按照 index 排序而非频数
  bar_value = df[col].value_counts().sort_index()
  # print(bar_value)
  if col != "uid":
    count += 1
    # 定位在一张大图上
    plt.subplot(3, 7, count)
    plt.title(col)
    # 调用 pandas 集成在内的方法使用 pyplot 画图
    bar_value.plot.bar()
    # 设置 x 轴字体大小，防止字体太大看不见
    plt.xticks(fontsize=6)

    for i in range(len(bar_value)):
      # 在柱状图上标明数据值
      plt.text(i, bar_value.values[i], bar_value.values[i],
               ha='center', va='bottom')
```

而对于值为数值类型的列，我们直接调用 `pyplot` 的 `hist` 方法进行绘制，并手动计算频数图的分段用以完整显示 x 轴的值。最后通过 `hist` 的返回值将频数图的数值标注在柱状图上方便于观看。

```python
elif not math.isnan(df[col].min()):
  # 将数据均分为 10 份，手动方便下方显示 x 轴的值
  min = df[col].min()
  max = df[col].max()

  list_bin = []

  if min == max:
    list_bin.append(min)
  else:
    step = (max - min) / 10

    for i in range(11):
      list_bin.append(round(min + i * step, 2))

  count += 1
  plt.subplot(3, 7, count)
  plt.title(col)
  # 调用 pyplot 画
  hist = plt.hist(df[col], edgecolor="black")
  # 自定义 x 轴的显示，默认显示不全
  plt.xticks(list_bin, rotation=45, fontsize=6)
  for i in range(len(hist[0])):
    plt.text(hist[1][i] + (hist[1][i+1] - hist[1][i])/2,
        hist[0][i], hist[0][i], ha='center', va='bottom')
```

整体上我们将所有列的图定位在一张大图上，并尽可能缩小间距和字体大小，以完整地展示：

```python
def showFig(df):
    # figure 新建一张图
    plt.figure()
    # 调节图像的位置，尽可能一张图放下
    plt.subplots_adjust(left=0.03, hspace=1, top=0.96, right=0.99, bottom=0.05)
    
    count = 0
    for col in df.columns:
      ...

    plt.show()
```

最终我们在对数据进行处理前和处理后分别进行一次可视化，以清晰地观察数据的分布形态和处理的效果，可视化图片如下：

![处理前](https://whale-picture.oss-cn-hangzhou.aliyuncs.com/img/before.png)

<p style="font-size: 14px; color: #808080; text-align: center">处理前</p>

![处理后](https://whale-picture.oss-cn-hangzhou.aliyuncs.com/img/after.png)

<p style="font-size: 14px; color: #808080; text-align: center">处理后</p>



## 3. 数据预处理

本次作业数据预处理部分我们分别进行了**缺失值处理、异常值处理、数据转换、数据归一化、数据标准化**

以上内容除了基本分外，还包括如下处理

- 缺失值处理选用机器学习模型进行处理
- 异常值处理将 <2% 和 >98% 的数据使用 2% 和 98% 的数据替换
- 使用数据归一化进行预处理，原因详见下方具体说明

这些内容请见下方详细说明，代码以星级评估为示例，信用评估与其类似

### 3.1 缺失值处理

#### 3.1.1 去除缺失值过多的列

代码如下所示

- `progress.start_progress("缺失值处理中")`, `progress.stop_progress()` 用于在控制台显示程序进行到哪一步以及消耗时间

```python
progress.start_progress("缺失值处理中")
# 计算文件每一列的缺失值比例并保存至 star_train_missing.csv 文件
star_train.isnull().mean().to_csv('star_result_data/star_train_missing.csv')
print("缺失比例已保存至 star_train_missing.csv")
# 去除缺失值比例大于 0.7 的列
star_train = star_train.loc[:, star_train.isnull().mean() < 0.7]
progress.stop_progress()
```

#### 3.1.2 填充缺失值

在一开始，我们小组对于数值型变量，用中位数填充缺失值；对于类别型变量，用众数填充缺失值

- `star_train.loc[:, numlist] = star_train[numlist].fillna(star_train[numlist].median())`
- `star_train.fillna(star_train.mode().iloc[0], inplace=True)`

但是这种操作会引入偏差，得到的结果不够精确，因此选择使用机器学习模型来填充缺失值

又由于训练数据中存在缺失值，因此选用 `HistGradientBoostingRegressor`、`HistGradientBoostingClassifier` 在训练时可以自动处理缺失值情况

代码如下所示，使用 Regressor 来进行填充数值型变量，使用 Classifier 来进行填充类别型变量

```python
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
```

### 3.2 异常值处理

对于异常值数据，我们小组将 <2% 和 >98% 的数据使用 2% 和 98% 的数据替换，代码如下，需要注意标签列不能处理

```python
progress.start_progress("异常值处理中")
# <2%和>98%的使用2%和98%的数据替换
star_train = pp.replace_data(star_train, ['star_level'])
progress.stop_progress()
```

```python
def replace_data(df, unused_col):
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and col not in unused_col:
            quantile_2 = df[col].quantile(0.02)
            quantile_98 = df[col].quantile(0.98)
            df.loc[df[col] < quantile_2, col] = quantile_2
            df.loc[df[col] > quantile_98, col] = quantile_98
    return df
```

### 3.3 数据转化

通过数据转化可以将特征编码，使用LabelEncoder等编码器可以将类别映射为整数，使得模型能够处理这些变量。这种转换可以帮助模型理解和利用类别信息，从而提高预测性能。

```python
progress.start_progress("正在将分类转换为数值类型")
le = LabelEncoder()
star_train[catelist] = star_train[catelist].apply(le.fit_transform)
star_train.to_csv('star_result_data/star_train_trans.csv', index=False)
```

### 3.4 数据归一化

数据预处理步骤中，我们除了给出的基本步骤，还对数据进行了归一化处理，理由如下：

- 在机器学习领域中，不同特征往往具有不同的量纲和量纲单位，这样的情况会影响到数据分析的结果，而**数据归一化可以将不同特征之间的数值范围调整到相同的尺度上**
- 数据归一化具有很多好处，包括：**消除特征之间的尺度差异、收敛速度加快、模型解释性增加、避免异常值对模型的影响**

代码如下所示

```python
progress.show_progress("数据归一化中")
scaler = MinMaxScaler()
star_train[colist] = scaler.fit_transform(star_train[colist])
```

### 3.5 数据标准化

数据标准化将数据转换为均值为 0、标准差为 1 的正态分布，使得数据更加符合统计假设和模型的要求；可以将数据的尺度转换为标准差单位，使得不同特征的权重更容易比较，代码如下所示

```python
progress.show_progress("数据标准化中")
scaler = StandardScaler()
star_train[colist] = scaler.fit_transform(star_train[colist])
star_train.to_csv('star_result_data/star_train_std.csv', index=False)
progress.stop_progress()
print("完成！请查看 star_train_std.csv")
```



## 4.特征工程

本次作业中我们利用变量**方差、枚举值占比、相关性、多重共线性**对特征进行选择，筛选出对预测有助的特征，减少冗余特征的影响。~~（尝试使用PCA进行数据降维，因降维后列不确定，模型使用难以适应，所以放弃）~~

### 4.1 连续型变量方差

方差是衡量连续型变量离散程度的度量，如果某个连续型变量的方差接近0，说明其特征值趋向于单一值的状态，对模型帮助不大，可剔除该变量。观察所选变量方差，将阈值定为1，剔除所有方差小于1的连续型变量。

代码如下所示：

```python
# 3.1 计算连续型变量的方差
# 计算变量方差并保存到 star_train_var.csv 文件
star_train[colist].var().to_csv('star_train_var.csv')
# 删除方差小于 1 的变量
var_to_drop = filter.get_low_var_cols(star_train[numlist])
colist = [i for i in colist if i not in var_to_drop]

def get_low_var_cols(data):
    var = data.var()
    to_delete = var[var < 1].index.tolist()
    return to_delete
```

### 4.2 类别型变量枚举值占比

如果某个类别型变量的枚举值样本量占比分布，集中在单一某枚举值上，说明其对模型影响不大，可剔除该变量。观察所选变量枚举值占比分布，将阈值定位0.9，剔除所有单一枚举值占比大于0.9的类型型变量。

代码如下所示：

```python
# 3.2 计算类别型变量枚举值占比
# 删除枚举值分布集中在单一枚举值上的变量
fre_to_drop = filter.get_single_enum_cols(star_train[catelist])
colist = [i for i in colist if i not in fre_to_drop]

def get_single_enum_cols(data):
    to_delete = []
    for col in data.columns:
        if len(data[col].value_counts()) > 1:
            value_counts = data[col].value_counts(normalize=True)
            if (value_counts >= 0.9).sum() > 0:
                to_delete.append(col)
    return to_delete
```

### 4.3 变量相关性

变量相关性是指两个或多个变量之间的关联程度。通常使用相关系数来度量变量之间的相关性。相关系数的取值范围是 -1 到 1，其中取值为 -1 表示完全负相关（一个变量增加时，另一个变量减少），0 表示无相关性，而取值为 1 表示完全正相关（一个变量增加时，另一个变量也增加）。高度相关的变量可能会导致模型不稳定，难以解释和过拟合等问题。因此对数据进行变量筛选，将高度相关的变量剔除掉，以提高模型的性能和可解释性。一般认为相关性大于0.7即说明变量之间高度相关，所以剔除相关性大于0.7的变量。

代码如下所示：

```python
# 3.3 计算变量相关性
# 计算变量相关性并保存至 star_train_corr.csv 文件
star_train[colist].corr().to_csv('star_train_corr.csv')
# 剔除相关性大于 0.7 的变量
corr_to_drop = filter.forward_delete_corr(star_train[colist])
colist = [i for i in colist if i not in corr_to_drop]

def forward_delete_corr(data):
    # 计算相关系数矩阵
    corr = data.corr().abs()
    # 选取相关系数矩阵的上三角部分
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    # 找出相关系数大于0.7的变量并添加到待删除列表中
    to_delete = [column for column in upper.columns if any(upper[column] > 0.7)]
```

### 4.4 多重共线性

多重共线性是一种数据分析中的问题，指在多元回归模型中，自变量之间存在高度相关性，导致模型参数的不确定性增加、预测结果不稳定或解释力下降等问题。使用VIF进行多重共线性检验，VIF测量了由于预测变量之间的共线，估计回归系数的方差增加了多少。VIF值为1表示没有多重共线性，而值高于1表示多重共线性水平增加，值高于5或10表示严重的多重共线性问题，所以剔除VIF值大于10的变量。

代码如下所示：

```python
# 3.4 计算多重共线性
# 计算多重共线性并剔除VIF大于 10 的变量
vif_to_drop = filter.get_low_vif_cols(star_train[colist])
colist = [i for i in colist if i not in vif_to_drop]
catelist = [i for i in catelist if i in colist]
numlist = [i for i in numlist if i in colist]

def get_low_vif_cols(data):
    to_delete = []
    # 循环剔除VIF值大于10的变量，直至所有变量的VIF值均小于10
    while True:
        vif = pd.DataFrame()
        vif["variables"] = data.columns
        vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        vif.to_csv('star_train_vif.csv')
        if vif["VIF"].max() > 10:
            # 找出VIF值最大的变量并删除
            col_to_drop = vif.loc[vif["VIF"].idxmax(), "variables"]
            to_delete.append(col_to_drop)
            data = data.drop(col_to_drop, axis=1)
        else:
            break
    return to_delete
```



## 5. 模型部分

本次作业我们先后使用了**逻辑回归、决策树、随机森林、XGBoost**算法模型，并依据**准确率、平衡准确率、精确率、召回率、F1分数、Cohen's Kappa系数、混淆矩阵**进行了评估，最终选择**星级评估使用XGBoost模型，信用评估使用随机森林模型**

代码以星级评估为例，信用评估与其类似

### 5.1 模型选择

#### 5.1.0 划分训练集和测试集

```python
# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(star_train[colist],
                                                    star_train['star_level'],
                                                    test_size=0.3,
                                                    random_state=0)
```

#### 5.1.1 逻辑回归

代码如下所示:

```python
# 训练模型
lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)
# 预测
y_pred = lr.predict(X_test)
```

为了更好地训练模型，将最大迭代次数设置为 500

#### 5.1.2 决策树

代码如下所示:

```python
# 训练模型
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
# 预测
y_pred = dt.predict(X_test)
```

#### 5.1.3 随机森林

代码如下所示:

```python
# 训练模型
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
# 预测
y_pred = rf.predict(X_test)
```

#### 5.1.4 XGBoost

代码如下所示:

```python
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
# 构建XGBoost分类器
xgb_clf = xgb.XGBClassifier()
# 训练XGBoost分类器
xgb_clf.fit(X_train, y_train)
# 预测测试集的标签
y_pred = xgb_clf.predict(X_test)
```

### 5.2 模型评估

#### 5.2.1 准确率、平衡准确率、精确率、召回率、F1分数、Kappa系数

代码如下所示

```python
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print('模型的准确率为：', accuracy)
# 计算平衡准确率为
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print('平衡准确率为:', balanced_acc)
# 计算精确率和召回率
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)  # 计算宏平均精确率
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)  # 计算宏平均召回率
print("精确率为: ", precision)
print("召回率为: ", recall)
# 计算F1分数
f1 = f1_score(y_test, y_pred, average='macro')  # 计算宏平均F1分数
print("F1分数为: ", f1)
# 计算Cohen's Kappa系数
kappa = cohen_kappa_score(y_test, y_pred)
print("Cohen's Kappa系数为: ", kappa)
```

##### 星级评估

| 模型     | 准确率 | 平衡准确率 | 精确率 | 召回率 | F1分数 | Cohen's Kappa系数 |
| -------- | ------ | ---------- | ------ | ------ | ------ | ----------------- |
| 逻辑回归 | 0.7472 | 0.3843     | 0.4253 | 0.3843 | 0.3966 | 0.5923            |
| 决策树   | 0.7698 | 0.6591     | 0.6743 | 0.6591 | 0.6660 | 0.6507            |
| 随机森林 | 0.8001 | 0.6309     | 0.7108 | 0.6309 | 0.6614 | 0.6945            |
| XGBoost  | 0.8102 | 0.6271     | 0.7488 | 0.6273 | 0.6649 | 0.7074            |

##### 信用评估

| 模型     | 准确率 | 平衡准确率 | 精确率 | 召回率 | F1分数 | Cohen's Kappa系数 |
| -------- | ------ | ---------- | ------ | ------ | ------ | ----------------- |
| 逻辑回归 | 0.6993 | 0.3937     | 0.5028 | 0.3937 | 0.3899 | 0.1651            |
| 决策树   | 0.8155 | 0.6860     | 0.7175 | 0.6860 | 0.6994 | 0.5881            |
| 随机森林 | 0.8502 | 0.7147     | 0.7835 | 0.7147 | 0.7424 | 0.6563            |
| XGBoost  | 0.8491 | 0.7073     | 0.7860 | 0.7073 | 0.7377 | 0.6525            |

#### 5.2.2 混淆矩阵

主函数中代码如下所示

```python
# 计算混淆矩阵并保存为图片
save_path = 'confusion_matrix.png'
class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
me.plot_confusion_matrix(y_test, y_pred, class_names, save_path)
```

plot_confusion_matrix的具体实现为

```python
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 绘制混淆矩阵图
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion matrix')
    plt.colorbar()

    # 设置x轴和y轴刻度及标签
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # 在矩阵中填充数字标签
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 保存混淆矩阵图
    plt.savefig(save_path)
```

使用XGBoost模型，生成的混淆矩阵如下图所示

<img src="https://whale-picture.oss-cn-hangzhou.aliyuncs.com/img/image-20230512003128866.png" alt="image-20230512003128866" style="zoom:80%;" />

### 5.3 模型应用

代码如下所示，最终预测结果见 star_test_xgb.csv 文件

由于使用XGBoost训练模型时，对数据进行了LabelEncoder处理，因此预测后需要映射回原有数据

```python
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
```

