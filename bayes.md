# 朴素贝叶斯分类器
[TOC]
## 简介
朴素贝叶斯分类器Demo
## 1. Demo1
根据身高、体重、脚掌大小预测性别。

样例参照网页：[https://wenku.baidu.com/view/654f1a100622192e453610661ed9ad51f01d54de.html](https://wenku.baidu.com/view/654f1a100622192e453610661ed9ad51f01d54de.html)

**样本：**

|性别|身高(英尺)|体重(磅)|脚掌(英寸)|
|-----|------------|---------|-----------|
|男|6| 180|12|
|男|5.92|190|11|
|男|5.58|170|12|
|男|5.92|165|10|
|女|5|100|6|
|女|5.5|150|8|
|女|5.42|130|7|
|女|5.75|150|9|

**问题：已知某人身高6英尺、体重130磅、脚掌8英寸，请问该人是男是女。**


**正态分布密度函数：** $ p(height|male)=\frac{1}{\sqrt{2\pi\sigma^2}} exp(\frac{-(身高-\mu)^2}{2\sigma^2}) \quad $

**贝叶斯分类器：**$P(性别|身高,体重,脚掌)=\frac{P(身高,体重,脚掌|性别) \times P(性别)}{P(身高,体重,脚掌)} \quad \quad \quad $

**因为 $P(身高,体重,脚掌)$ 是常数项，所以只需要比较 $ P(身高,体重,脚掌|性别) \times P(性别) $ 的大小即可。**
$P(身高,体重,脚掌大小|性别) \times P(性别)=  P(身高|性别) \times P(体重|性别) \times P(脚掌大小|性别) \times P(性别)$

``` python
# -- coding: utf-8 --
import numpy as np

'''
此demo需求为：通过身高、体重、脚掌大小来预测性别

'''


# 创建数据
def createData():
    array_male = np.array([[6, 180, 12], [5.92, 190, 11], [5.58, 170, 12], [5.92, 165, 10]])
    array_female = np.array([[5, 100, 6], [5.5, 150, 8], [5.42, 130, 7], [5.75, 150, 9]])
    return array_male, array_female


# 得到平均数与样本方差
def mean_and_var_function(array):
    mean = np.mean(array)
    var = np.var(array, ddof=1)
    return mean, var


# 计算正态分布密度函数
def probability_density_function(mean, var, x):
    step1 = (1.0 / ((2 * np.pi * var) ** 0.5))
    step2 = np.exp((-(x - mean) ** 2) / (2 * var))
    return step1 * step2


# 贝叶斯算法
def bayes(array, arraydata, p):
    height_mean, height_var = mean_and_var_function(array[:, 0])
    weight_mean, weight_var = mean_and_var_function(array[:, 1])
    size_mean, size_var = mean_and_var_function(array[:, 2])

    p_height = probability_density_function(height_mean, height_var, arraydata[0])
    p_weight = probability_density_function(weight_mean, weight_var, arraydata[1])
    p_size = probability_density_function(size_mean, size_var, arraydata[2])

    probability = p_height * p_weight * p_size * 0.5
    return probability


# 主函数
def dorun():
    array_male, array_female = createData()
    p_male = bayes(array_male, [6, 130, 8], 0.5)
    p_female = bayes(array_female, [6, 130, 8], 0.5)
    print('probability of male is [%e]' % p_male)
    print('probability of female is [%e]' % p_female)
    if (p_male > p_female):
        print("is male")
    elif (p_male < p_female):
        print("is female")

dorun()
```

## 2. Demo2
 **约会网站数据分类**
### 2.1 数据集：总体数据共1000条。数据文件为`datingTestSet2.txt`

|玩视频游戏所消耗时间百分比|每年获得的飞行常客里程数|每周消费冰激淋公升数|样本分类|
|---------------------------------|-------------------------------|--------------------------|----------|
|40920|8.326976|0.953952|3|
|14488|7.153469|1.673904|2|
|26052|1.441871|0.805124|1|
|75136|13.147394|0.428964|1|
|38344|1.669788|0.134296|1|
|72993|10.141740|1.032955|1|
|35948|6.830792|1.213192|3|
|42666|13.276369|0.543880|3|
|67497|8.631577|0.749278|1|
|....|....|....|....|

**样本分类：**
* 1:不喜欢
* 2:一般
* 3:非常喜欢

### 2.2 样本数据与测试数据
样本数据集：总体数据中后900条。
测试数据集：总体数据前100条。

### 2.3 实现代码：
```python
# -- coding: utf-8 --
import numpy as np

'''
贝叶斯计算约会网站分类
'''


# 读取文件，将文件内容转化为矩阵
def createData():
    fr = open('data/datingTestSet2.txt')
    arraylines = fr.readlines()
    numberOfLines = len(arraylines)
    returnMat = np.zeros((numberOfLines, 4))

    index = 0
    for line in arraylines:
        line = line.strip()
        listFromLine_tmp = line.split('\t')
        listFromLine = np.array([float(x) for x in listFromLine_tmp])
        returnMat[index, :] = listFromLine[0:4]
        index += 1

    return returnMat, numberOfLines


# 计算平均值与样本方差
def mean_and_var(array):
    mean = np.mean(array.tolist())
    var = np.var(array.tolist(), ddof=1)
    return mean, var


# 计算每个类型出现的概率
def probability_of_type(array1, array2, array3, line_num):
    return len(array1) / line_num, len(array2) / line_num, len(array3) / line_num


# 计算正态分布密度函数
def probability_density_function(mean, var, x):
    step1 = (1.0 / ((2 * np.pi * var) ** 0.5))
    step2 = np.exp((-(x - mean) ** 2) / (2 * var))
    return step1 * step2


# 贝叶斯算法
def bayes(returnMat, arraydata, p):
    mean_fly, var_fly = mean_and_var(returnMat[:, 0])
    mean_play, var_play = mean_and_var(returnMat[:, 1])
    mean_ice, var_ice = mean_and_var(returnMat[:, 2])

    p_fly = probability_density_function(mean_fly, var_fly, arraydata[0])
    p_play = probability_density_function(mean_play, var_play, arraydata[1])
    p_ice = probability_density_function(mean_ice, var_ice, arraydata[2])

    probability = p_fly * p_play * p_ice * p
    return probability


# 对样本数据根据喜好程度进行分组，得到不同分类的所有矩阵数据
def classificationData(array):
    array_tmp = array[:, -1];
    class1 = np.sum(array_tmp == 1)
    class2 = np.sum(array_tmp == 2)
    class3 = np.sum(array_tmp == 3)

    class1_data = np.zeros((class1, 3))
    class2_data = np.zeros((class2, 3))
    class3_data = np.zeros((class3, 3))
    index_class1 = 0
    index_class2 = 0
    index_class3 = 0

    for line in array:
        if (line[-1] == 1):
            class1_data[index_class1, :] = line[0:3]
            index_class1 += 1
        elif (line[-1] == 2):
            class2_data[index_class2, :] = line[0:3]
            index_class2 += 1
        elif (line[-1] == 3):
            class3_data[index_class3, :] = line[0:3]
            index_class3 += 1
    return class1_data, class2_data, class3_data


# 调用函数
def dorun():
    returnMat, line_num = createData()  # 得到所有的样例数据
    sample_data_lines = round(line_num * 0.9)  # 将总体数据中的90%的数据作为样本数据集
    sample_data = returnMat[line_num - sample_data_lines:]  # 得到样本数据集

    class1, class2, class3 = classificationData(sample_data)  # 对各种分类数据进行分类
    p1, p2, p3 = probability_of_type(class1, class2, class3, line_num)  # 计算个分类数据的概率
    test_data = returnMat[0:line_num - sample_data_lines]  # 将总体数据中的10%做为测试数据集
    test_line = np.shape(test_data)[0]  # 得到测试数据集的条数

    for i in range(test_line):
        testdata = test_data[i]

        a = []
        p_type_1 = bayes(class1, testdata, p1)  # 计算类型1的概率
        a.append(p_type_1)
        p_type_2 = bayes(class2, testdata, p2)  # 计算类型3的概率
        a.append(p_type_2)
        p_type_3 = bayes(class3, testdata, p3)  # 计算类型3的概率
        a.append(p_type_3)

        max = np.max(a)  # 得到概率最大的类型
        
        # 数据结果：通过计算得到的分类结果-文件中真实的分类结果
        if (max == p_type_1):
            print("type is 1 , real type is =%s" % int(testdata[3]))
        elif (max == p_type_2):
            print("type is 2 , real type is =%s" % int(testdata[3]))
        elif (max == p_type_3):
            print("type is 3 , real type is =%s" % int(testdata[3]))

dorun()

```










