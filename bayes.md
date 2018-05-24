#朴素贝叶斯分类器
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




