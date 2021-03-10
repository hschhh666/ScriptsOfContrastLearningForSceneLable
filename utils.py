import numpy as np
from cv2 import cv2
import re
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn import decomposition
import matplotlib
import matplotlib.image as gImage
from sklearn.manifold import TSNE
from matplotlib.ticker import FuncFormatter
import scipy.stats
import time
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import os

def calculateDis(GNSS,posA,posB):# posA、B是两个位置的索引，返回这两个位置之间的直线距离和累计距离
    res1 = np.sqrt(np.square(GNSS[posA][0]-GNSS[posB][0])+ np.square(GNSS[posA][1]-GNSS[posB][1]))
    res2 = 0
    for i in range(posA+1,posB+1):
        res2 += np.sqrt(np.square(GNSS[i][0]-GNSS[i-1][0])+ np.square(GNSS[i][1]-GNSS[i-1][1]))
    return res1, res2 

def calculateAngel(GNSS,posA,posB):# # posA、B是两个位置的索引，返回这两个位置之间的方向差，范围为[0,180]
    lenth = len(GNSS) - 1
    if posA <= 0 or posA >= lenth or posB <= 0 or posB >= lenth:
        return 0
    angleA = GNSS[posA][2]
    angleB = GNSS[posB][2]
    delta = abs(angleA - angleB)
    return 360 - delta if delta > 180 else delta


def getAnchorPosNegIdx(path): # path是数据集路径，返回的是每个锚点及其对应正/负样本的id，数据组织形式是[[anchor1,[pos1,pos2,...]],[anchor2,[pos1,pos2,...]]...]
    dirs = []
    for _, dirs, _ in os.walk(path):
        break
    dirs = [int(i) for i in dirs]
    dirs = sorted(dirs)
    anchor_pos_list = []
    for d in dirs:
        cur_anchor_pos_list = []
        cur_anchor_pos_list.append(d)
        files = []
        for _, _, files in os.walk(os.path.join(path, str(d))):
            break
        pos_list = []
        for f in files:
            match = re.match('(\d*)\.png',f)
            f = int(match.group(1))
            pos_list.append(f)
        cur_anchor_pos_list.append(pos_list)
        anchor_pos_list.append(cur_anchor_pos_list)
    
    anchor_neg_list = []
    anchor_num = len(anchor_pos_list)
    for i in range(anchor_num):
        cur_anchor_neg_list = [anchor_pos_list[i][0]]
        neg_list = []
        if i == 0:
            neg_list = anchor_pos_list[1][1] + anchor_pos_list[anchor_num-1][1]
        elif i == anchor_num - 1:
            neg_list = anchor_pos_list[anchor_num - 2][1] + anchor_pos_list[0][1]
        else:
            neg_list = anchor_pos_list[i-1][1] + anchor_pos_list[i+1][1]
        cur_anchor_neg_list.append(neg_list)
        anchor_neg_list.append(cur_anchor_neg_list)

    return anchor_pos_list, anchor_neg_list

def getAnchorPosNegIdx2(path, frame_count):
    files = []
    for _,_,files in os.walk(path):
        pass
    imgNamesInt = [int(i[0:10]) for i in files]
    imgNamesInt = sorted(imgNamesInt)
    anchor_pos_list = []
    sampleNum = 16
    for i, _ in enumerate(imgNamesInt):
        if i == len(imgNamesInt) - 1:
            break
        cur_anchor_pos_list = []
        cur_anchor_pos_list.append(imgNamesInt[i])
        pos_list = list(np.linspace(imgNamesInt[i], imgNamesInt[i+1], sampleNum + 1).astype('int32'))[0:-1]
        cur_anchor_pos_list.append(pos_list)
        anchor_pos_list.append(cur_anchor_pos_list)
    
    cur_anchor_pos_list = []
    cur_anchor_pos_list.append(imgNamesInt[-1])
    tmp_idx = list(range(imgNamesInt[-1], frame_count)) + list(range(0, imgNamesInt[0]))
    pos_list = tmp_idx[0:len(tmp_idx):int(len(tmp_idx)/sampleNum)]
    cur_anchor_pos_list.append(pos_list)
    anchor_pos_list.append(cur_anchor_pos_list)


    anchor_neg_list = []
    anchor_num = len(anchor_pos_list)
    for i in range(anchor_num):
        cur_anchor_neg_list = [anchor_pos_list[i][0]]
        neg_list = []
        if i == 0:
            neg_list = anchor_pos_list[1][1] + anchor_pos_list[anchor_num-1][1]
        elif i == anchor_num - 1:
            neg_list = anchor_pos_list[anchor_num - 2][1] + anchor_pos_list[0][1]
        else:
            neg_list = anchor_pos_list[i-1][1] + anchor_pos_list[i+1][1]
        cur_anchor_neg_list.append(neg_list)
        anchor_neg_list.append(cur_anchor_neg_list)
    return anchor_pos_list, anchor_neg_list

        
def JS_D(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)


def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure()

    # plt.figure(figsize=(12, 8), dpi=100)
    # np.set_printoptions(precision=2)

    size = len(classes)
    acc_matrix = np.zeros_like(cm, dtype=float)
    for i in range(size):
        sums = np.sum(cm[i],dtype=float)
        for j in range(size):
            acc_matrix[i][j] = (cm[i][j] / sums)*100.0

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        acc = acc_matrix[y_val][x_val]
        plt.text(x_val, y_val, "%d\n(%.1f%%)" % (c,acc), color='red', va='center', ha='center')
    

        # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    # plt.gcf().subplots_adjust(bottom=0.15)
    
    xlocations = range(len(classes))
    plt.xticks(xlocations, classes)
    plt.yticks(xlocations, classes)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()

    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    

    
    # show confusion matrix
    # plt.show()


if __name__ == '__main__':
    posFile = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\pos.npy' # 把字幕文件里的经纬度提取出来保存成numpy数据，这样方便读取节约时间
    GNSS = np.load(posFile)
    dis1,dis2 = calculateDis(GNSS, 1200,1800)
    print(dis1,dis2)

    datasetPath = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\labeledData\\test2\\train'
    anchor_pos_list = getAnchorPosNegIdx(datasetPath)

    print()
