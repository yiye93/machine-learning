import math
import pandas as pd

csv_file = "./tq.csv"
# 正例的标识(对应csv的class)
p_flag = "p"
# 反例的标识(对应csv的class)
n_flag = "n"

# 这里读取一个csv文件，每个特征以,分割，最后一例是标签例子，即正例或者反例
def loadDataSet(csvfile: str):
    """
    导入数据
    @ return dataSet: 读取的数据集
    """
    # 对数据进行处理
    dataSet = pd.read_csv(csvfile, delimiter=',')
    # 读取出所有的特征列，包含标签(正反例)
    feature_labels = list(dataSet.columns.values)
    # 读取出数据值
    dataSet = dataSet.values
    return dataSet, feature_labels

def getOneFeatureAttrs(dataSet: list, feature_idx: int):
    """
    获取一个特征的所有属性值
    @ param dataSet: 数据集
    @ param feature: 指定的一个特征(这里是用下标0，1，2..表示)
    """
    attrs = [data[feature_idx] for data in dataSet]
    return set(attrs)

def calcIpn(p:int, n: int):
    """
    按书上公示计算I(p,n)
    :param p: 正例个数
    :param n: 反例个数
    :return:
    I(p, n)的结果
    """
    # 注意: p, n为0的特殊情况
    return -(p/(p+n))*math.log2(p/(p+n))-(n/(p+n))*math.log2(n/(p+n)) if p != 0 and n != 0 else 0

def getDatasetPN(dataSet: list):
    """
    获取数据集的p, n值
    :param dataSet: 数据集
    :return:
    """
    p, n = 0, 0
    for data in dataSet:
        label = data[-1]
        if label == p_flag:
            p += 1
        elif label == n_flag:
            n += 1
    return p, n

def getOneFeatureAttrPN(dataSet: list, feature_idx: int, attr: str):
    """
    计算一个特征对应的属性的p, n值
    :param dataSet: 数据集
    :param feature_idx: 指定的一个特征(这里是用下标0，1，2..表示)
    :param attr: 属性名称, 如sunny, windy...
    :return:
    """
    p = 0
    n = 0
    for data in dataSet:
        if data[feature_idx] == attr:
            # 类别即标签值,为数组的最后一列
            label = data[-1]
            if label == p_flag:
                p += 1
            elif label == n_flag:
                n += 1
    return p, n

def calcOneFeatureEa(dataSet: list, feature_idx: int):
    """
    获取一个特征的E(A)值
    :param dataSet: 数据集
    :param feature_idx: 指定的一个特征(这里是用下标0，1，2..表示)
    :return:
    """
    attrs = getOneFeatureAttrs(dataSet, feature_idx)
    # 获取数据集的p, n值
    p, n = getDatasetPN(dataSet)
    ea = 0.0
    for attr in attrs:
        # 获取每个属性值对应的p, n值
        attrP, attrN = getOneFeatureAttrPN(dataSet, feature_idx, attr)
        # 计算属性对应的ipn
        attrIPN = calcIpn(attrP, attrN)
        ea += (attrP+attrN)/(p+n) * attrIPN
    return ea

def calcOneFeatureGain(dataSet: list, feature_idx: int):
    """
    获取一个特征的信息增益
    :param dataSet: 数据集
    :param feature_idx: 数据集
    :return:
    """
    # Gain(A) = I(p,n) - E(A)
    p, n = getDatasetPN(dataSet)
    ipn = calcIpn(p, n)
    gain = ipn - calcOneFeatureEa(dataSet, feature_idx)
    return gain

def selectBestFeature(dataSet: list):
    """
    计算信息增益，挑选最好的特征值
    :param dataSet: 数据集
    :return:
    """
    # 不使用最后一列标签列
    best_feature_idx = 0
    gain = float('-inf')
    for feature_idx in range(len(dataSet[0])-1):
        gain_tmp = calcOneFeatureGain(dataSet, feature_idx)
        if gain_tmp > gain:
            best_feature_idx = feature_idx
            gain = gain_tmp
    return best_feature_idx

def splitDataByFeatureAttr(dataSet: list, feature_idx: int, attr: str):
    """
    获取一个相关特征属性值的样本集合。如获取所有outlook特征是sunny的样本数(即csv的行)
    :param dataSet: 数据集
    :param feature_idx: 指定的一个特征(这里是用下标0，1，2..表示)
    :param attr: 属性名
    :return:
    """
    retData = []
    for data in dataSet:
        if data[feature_idx] == attr:
            retData.append(data)
    return retData

def createId3Tree(dataSet: list, feature_labels: list):
    """
    递归构建一颗id3的决策树
    :param dataSet: 数据集
    :return:
    """
    # 终止条件: 数据集里只有一种样例的时候,正样例or反样例

    # 获取该数据集的所有样例属性值集合(即最后一列的class类的p, n集合)
    classList = [data[-1] for data in dataSet]
    # 当只有正列或者反例的时候，停止
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 通过信息增益，计算获取最好的特征值
    best_feature_idx = selectBestFeature(dataSet)
    best_feature_label = feature_labels[best_feature_idx]
    myTree = {best_feature_label: {}}

    # 开启按该特征的所有属性值(如outlook对应的sunny,rain, overcast）重新分叉生成新的树
    # 可以对照示意图看
    attrs = getOneFeatureAttrs(dataSet, best_feature_idx)
    for attr in attrs:
        splitDataset = splitDataByFeatureAttr(dataSet, best_feature_idx, attr)
        splitLabel = feature_labels[:]
        myTree[best_feature_label][attr] = createId3Tree(splitDataset, splitLabel)
    return myTree

def predict(tree: dict, data: list):
    """
    输入一个数值, 通过id3决策树进行预测
    :param tree: 构建的id3决策树
    :param data: 输入的一行样本数据,如:["sunny", "high"]
    :return:
    """

import matplotlib.pyplot as plt
# 定义文本框和箭头格式
decisionNode = dict(boxstyle="round4", color='#3366FF')  #定义判断结点形态
leafNode = dict(boxstyle="circle", color='#FF6633')  #定义叶结点形态
arrow_args = dict(arrowstyle="<-", color='g')  #定义箭头

#绘制带箭头的注释
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
#计算叶结点数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


#计算树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


#在父子结点间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)  #在父子结点间填充文本信息
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  #绘制带箭头的注释
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

if __name__ == "__main__":
    dataSet, feature_labels = loadDataSet(csv_file)
    myTree = createId3Tree(dataSet, feature_labels)
    print(f"id3 tree is:{myTree}")
    createPlot(myTree)
