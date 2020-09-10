import numpy as np
import os
import re

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




if __name__ == '__main__':
    posFile = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\pos.npy' # 把字幕文件里的经纬度提取出来保存成numpy数据，这样方便读取节约时间
    GNSS = np.load(posFile)
    dis1,dis2 = calculateDis(GNSS, 1200,1800)
    print(dis1,dis2)

    datasetPath = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\labeledData\\test2\\train'
    anchor_pos_list = getAnchorPosNegIdx(datasetPath)

    print()
