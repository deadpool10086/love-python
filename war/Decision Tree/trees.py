from math import log
import matplotlib.pyplot as plt
import operator

decisionNode = dict(boxstyle = "sawtooth", fc="0.8")
leafNode = dict(boxstyle = "round4", fc="0.8")
arrow_args = dict(arrowstyle = "<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.axl.annotate(nodeTxt, xy=parentPt, xycoords= 'axes fraction',
     xytext = centerPt, textcoords='axes fraction',
     va = "center", ha="center", bbox=nodeType, arrowprops = arrow_args)
def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.axl = plt.subplot(111, frameon=False)
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt
def createDataSet():
    dataSet = [[1,1,'yes'], [1, 1, 'yes'], [1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
def splitDataSet(dataSet, axis, value):  #提取出dataSet中下表为axis号特征值为value的数据
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  #特征数
    baseEntropy = calcShannonEnt(dataSet)  #计算全部的香农熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] #提取所有数据的第一个特征的值
        uniqueVals = set(featList)   #转化为一个集合，即去除重复部分
        newEntropy = 0.0
        for value in uniqueVals:  #计算出按照第i个特征分割得到的香农熵
            subDataSet = splitDataSet(dataSet, i, value)  #划分出值为value的子集
            prob = len(subDataSet) / float(len(dataSet)) #计算出子集的权重
            newEntropy += prob * calcShannonEnt(subDataSet) #按照相应的权重乘以香农熵最后相加即为所求
        infoGain = baseEntropy - newEntropy  #求出划分后的和未划分的香农熵差值
        if infoGain > bestInfoGain:  #选出差值最大的作为划分特征
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
def maiorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount: classCoutn[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),
     key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): #如果只剩下一种结果就直接返回该结果
        return classList[0]
    if len(dataSet[0]) == 1: #如果没有剩下的特征作为分割直接返回种类最多的结果
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) #选择分割后的香农熵最低的特征分割
    # print(bestFeat)
    bestFeatLabel = labels[bestFeat]  #取得特征名字
    print(bestFeatLabel)
    myTree = {bestFeatLabel:{}}  #创建相应的字典
    del(labels[bestFeat]) #在特征名字列表中删除已取得的特征
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)  #取得该特征的所有值
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat,value),
         subLabels)
    return myTree

def getNumLeafs(myTree):
    numLeafs = 0;
#    print(myTree)
    firstStr = list(myTree.keys())[0]
#    print(firstStr)
    secondDict = myTree[firstStr]
#    print(secondDict)
#    input();
    for key in secondDict.keys():       #计数如果是字典就进去，不是就+1
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else: numLeafs += 1
    return numLeafs
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else: thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth
def retrieveTree(i):
    listOfTrees = [{'no surfacing':{0:'no',1:{'flippers': {0:'no', 1:'yes'}}}},\
    {'no surfacing': {0:'no',1:{'flippers': \
    {0:{'head':{0:'no',1:'yes'}}, 1:'no'}}}}]
    return listOfTrees[i]
def plotMidText(cntrPt, parentPt, textString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.axl.text(xMid, yMid, textString)
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, \
        plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff),
                cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.axl = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree, fw)
    fw.close()
def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)
