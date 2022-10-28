# VLong-
# 代码开源
由于便于可读性，代码是*.ipynb。可读性极强，运行结果也有，希望大家可以star and fork ，非常感谢。
链接: 
# 比赛介绍
汽车产业正在经历巨大变革，新能源汽车市场规模持续扩大，电池安全问题日益引发重视。 电池异常检测面临着汽车数据质量差，检出率低，误报率高，大量无效报警无法直接自动化运维等问题。

为了更好的检验电池安全问题，比赛通过募集优秀异常检测方案，使用特征提取、参数优化、异常对比等手段对实车数据进行处理，优化异常检测结果，以便更好的应用于车辆预警、故障模式识别等多种场景。

# 赛题简介

## 1、赛题链接

```
https://aistudio.baidu.com/aistudio/competition/detail/495/0/introduction
```
## 2、赛题背景
新能源车辆电池的故障检测对于及时发现车辆问题、排除隐患、保护人的生命财产安全有着重要意义。新能源电池的故障是多种多样的，包括热失控、析锂、漏液等，本次比赛数据中包含了多种故障类型，但在数据中统一标注为故障标签“1”，不做进一步的区分。

一般故障检测都会面临故障标签少的问题，在本次比赛中，我们对数据进行了筛选，正常数据和故障数据的比例不是非常悬殊，即便如此，常规的异常检测算法依然会是一个非常好的选择。

电池的数据是一种时序数据，在数据中的‘timestamp’列是该数据的时间戳，理解时序数据的处理方式可能会对异常检测有更好的效果。除此之外，发生异常的数据可能看起来很“正常”，需要提取更多的特征来进行分析。

## 3、比赛任务
**具体任务：**
本次赛题任务为“挑选出异常的电池数据”，选手在训练集对数据进行分析，选取特征，训练并优化模型等，最终需要检出测试集中的异常充电数据，保证较高的检出和较低的误报，评分会使用综合指标AUC。
## 4、评估指标
AUC（area under curve)

## 5、数据集
训练集：
文件类型为.pkl文件，每个pkl文件内容为元组形式（data,metadata）；
data：形状为（256，8），每列数据对应特征[‘volt’,‘current’,‘soc’,‘max_single_volt’,‘min_single_volt’,‘max_temp’,‘min_temp’,‘timestamp’]
metadata：包含label和mileage信息，label标签中‘00’表示正常片段，‘10’表示异常片段。

测试集：
文件类型为.pkl文件，每个pkl文件内容为元组形式（data,metadata）；
data：形状为（256，8），每列数据对应特征[‘volt’,‘current’,‘soc’,‘max_single_volt’,‘min_single_volt’,‘max_temp’,‘min_temp’,‘timestamp’]
metadata：仅包含mileage信息。

# 我就不啰里八嗦了
下面是我们的思路总结，代码部分在github，是一个*.ipynb文件，跑动的过程我们已经展示了，部分解释也写了。应该很容易懂，所以下面是想法的来源和分析。
## 赛题分析
首先，很明显是一个二分类问题，但是赛题给出的数据集数据极其不平衡比例大约是(neg)4:1(pos)。对于这种电池异常背景很显然异常状态有一些特征会很特别，所以入手思路基本就两个要么**搞数据**要么**搞特征**。下面从两个方面分析我们的做法。

## 特征方面
我们组也不是很懂这个电池这个玩意，特征没有做特别深入的分析和构建。特征构建方面很简单：由于是一条序列数据所以会对每一列的特征做**最大值、最小值、均值、方差、众数或者中位数、求和**。另外还有一些**相对的信息**，比如电压上一时刻减去下一时刻的相对电压信息，相对最大温度，相对最大单体电压、相对最小单体电压、温度缩放、以及最后一个时刻减去第一个时刻的变化。总体没什么特别的特征。

```python
def helper(x):
    '''
    x: 特征 
    比如 volt，会有序列的信息 比如第一时刻到256时刻的volt数据
    对这序列数据取到 最大值、最小值、均值、求和、median、方差
    这些特征用来反映这条时序数据的信息
    
    
    后续会对每一个特征都会进行这样的操作，另外还会引入新的特征序列，比如相对volt时序信息等等
    '''
    if isinstance(x, type(np.array([1,2,]))):
        x = pd.DataFrame(x)
    max_num = x.max()
    min_num = x.min()
    mean_num = x.mean()
    sum_num = x.sum()
    median_num = float(x.median())
    std_num = float(x.std())

    res = []
    for i in [max_num, min_num, mean_num, sum_num, median_num, std_num]:
        res.append(float(i))
    return res
```

```python
def get_features(m, ):
    '''
    m: 表示每一条数据  序列长度*['volt','current','soc','max_single_volt','min_single_volt','max_temp','min_temp','timestamp', 'mileage']
    
    以下的思路是先对每一个已有的序列特征根据helper提取到他的一些标量化特征来表示这条序列特征的信息
    '''
    
    # 所有的特征信息
    features = []  
    # 遍历已有的特征，注意这里比赛事给的特征多了一个 mileage
    for i in columns:
        l = helper(m[i])  # 对每一个序列特征提取他的标量特征
        features += l     # 把它加到features
        
    # 以下是相对volt信息，也就是上一时刻的volt减去前一时刻的volt，构造的新的特征信息，这里做了一定的扩大用来放大特征
    # 这里基本上就是假设有1-256 时刻 就是。2-256的特征减去 1-255
    one_last_volt = np.array(m['volt'][1:])*10
    zero_last_volt = np.array(m['volt'][0:-1])*10
    new_volt = one_last_volt - zero_last_volt
    new_volt_rel = np.hstack([new_volt, np.array(m['volt'].iloc[-1]) - np.array(m['volt'].iloc[0])])
    # 相对信息的最大最小 mean/sum/median
    features+=helper(new_volt_rel)  # 对序列特征进行标量化处理
    
    # 同理这里是max_single_volt的相对特征，做了缩放
    one_last_volt = np.array(m['max_single_volt'][1:])*1000
    zero_last_volt = np.array(m['max_single_volt'][0:-1])*1000
    new_volt = one_last_volt - zero_last_volt
    new_max_single_volt = np.hstack([new_volt,np.array(m['max_single_volt'].iloc[-1]) - np.array(m['max_single_volt'].iloc[0])])
    features+=helper(new_max_single_volt)
    
    # 同理这里是min_single_volt的相对特征，做了缩放
    one_last_volt = np.array(m['min_single_volt'][1:])*1000
    zero_last_volt = np.array(m['min_single_volt'][0:-1])*1000
    new_volt = one_last_volt - zero_last_volt
    new_min_single_volt = np.hstack([new_volt,np.array(m['min_single_volt'].iloc[-1]) - np.array(m['min_single_volt'].iloc[0])])
    features+=helper(new_min_single_volt)
    
    # 这里对温度做了缩放信息
    new_max_temp_vari = m['max_temp'] - 200
    new_min_temp_vari = m['min_temp'] - 160
    features += helper(new_max_temp_vari)
    features += helper(new_min_temp_vari)
    
    # 这里是特征变化的信息 也就是最后一个时刻和第一个时刻的每一个特征的变化特征。
    last_one_diff = list(m[columns].iloc[-1] - m[columns].iloc[0])
    features += last_one_diff
    
    # 最大volt和最小volt的差特征
    max_min_volt_change = m['max_single_volt'] - m['min_single_volt']
    features += helper(max_min_volt_change)
    # 最大温度和最小温度的差特征
    max_min_temp_change = m['max_temp'] - m['min_temp']
    features += helper(max_min_temp_change)
    
    return features
```
每一条数据在进行上面特征操作之后会构建出105个特征信息。具体可参考代码里面的运行过程

## 数据增强操作
数据增强应该算我们的**核心贡献点**。由于赛题给出的数据比例不均衡，尤其是在二分类条件下，深度学习模型肯定是不太可以的。想用的话肯定得直接增加数据。

给定的数据是1-256条序列数据。为了扩充数据，需要对数据做切分，将一条数据变成多条数据，同时保证数据的时序性。
**我们按照将一条数据划分成4条数据。**

由于是时序数据，所以取间断时刻构成的数据也算时序数据，而且出现异常的状态肯定是一段时间的所以间断时刻不会改变它是异常的本质。这个思路可以有一个简单的案例帮助大家感受一下，一个图片压缩实例：

比如现在有12条数据。我们按照每4个时刻为间断构建一条新的数据，具体如下：
```python
[1,2,3,4,5,6,7,8,9,10,11,12]. # 假设一条原始数据有12个时刻
# 拆成两条
[1, 3, 5, 7, 9, 11] # 奇数时刻
[2, 4, 6 ,8 ,10, 12] # 偶数时刻
# 思路扩展可以拆成4条
[0, 4, 8]
[1, 5, 9]
[2, 6, 10]
[3, 7, 11]
# 当然也可以拆成8条，我们尝试了效果不好，
# 可能是因为有的异常时段间隔小于8，导致拆成8条有的数据里面由原来的
# 异常变成了正常。这部分也没具体操作和试，或许也可以
```
这样一条时序数据被划分成了4条新的时序数据，同样的1-256会被划分成4条新的数据。因此每一条数据会被扩充4倍，

```python
new_df = []   # 存储结果
columns = ['volt','current','soc','max_single_volt','min_single_volt','max_temp',
           'min_temp','timestamp', 'mileage',]   # 这里就是需要处理的序列特征多了一个 mileage
for idx, m in tqdm(enumerate(group_data)):  # 之前按照每条数据对应的索引进行了分组，现在遍历每一条数据
    for i in range(4):                      # 这里是我们核心的想法，参考上面的描述
        df = m.iloc[i::4]
        features = get_features(df)         # 特征处理
        temp_df = pd.DataFrame(features).T  # 将其保存
        temp_df['label'] = 1                # 打标签
        new_df.append(temp_df)
```


## 实施方面
在进行特征处理以及数据增强之后，接下来模型方面我们采用**LightGBM**，由于该模型是对特征极其敏感的。所以我们在对每一条数据进行特征处理之后会把之前的8个特征扩充成105条特征。105条全部用于模型的话效果很是差劲，我们分析可能某一些特征是无用或者重复的，所以对105个特征进行可视化处理，对比正负样本的差异，寻找能够明显分类正负样本的特征，进而**人工选择**出了大约30多个最佳的特征(可能会有更好的组合，我们没有太多的尝试)。
### 模型部分
```python
# 基础模型选择为LGBMClassifier
model5 = LGBMClassifier(boosting_type='rf', learning_rate=0.0001,max_depth=6,num_leaves=2**6,
                       reg_lambda=0.02,random_state=2, is_unbalance=False,
                       bagging_freq=10,
                       bagging_fraction= 0.5,
                       feature_fraction=0.5,
                       reg_alpha=0.01)
model5.fit(x, y)
pred5_score = model5.predict_proba(test_x)[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred5_score, pos_label=1)
print('model5：\n', model5, '\nauc:', metrics.auc(fpr, tpr))

# 正确的预测结果auc: 0.9987597833542866，如果不对可能导致A榜结果预测不准确，
# 处理办法可以参考我们提供已经处理好的数据，或者将LGBMClassifier包升级 满足版本要求
```
### 测试部分
由于测试数据也会被分成4组所以预测结果是4组的平均值或者最后一组的信息，这里其实还有很多思路操作空间，我们没有尝试

```python
# 计算A榜的预测结果，将4组预测结果取平均值或者最后一组
def compute_test():
    score = 0
    for i in test_df:
        score = model5.predict_proba(i.iloc[:, features_index])[:, 1]
    
    # score /= 4
    submision_df = pd.DataFrame()
    submision_df['file_name'] = test_df1['pkl'].apply(lambda x:x[7:])
    submision_df['score'] = score
    submision_df.to_csv('submision.csv', index=False)
    return submision_df
```
