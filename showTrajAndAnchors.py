import numpy as np
from cv2 import cv2
import re
import matplotlib.pyplot as plt
from utils import *
import random
from sklearn.cluster import KMeans
from sklearn import decomposition
import matplotlib
import matplotlib.image as gImage
from sklearn.manifold import TSNE
from matplotlib.ticker import FuncFormatter


datasetPath = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/RingroadTestData/labeledData/CampusSceneDataset_labeldByTurnAndDynamicTraffic/anchorImgs'
subtitleFile = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\video_subtitle.srt' # 字幕文件，里面保存了每帧图片对应的经纬度
posFile = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\pos.npy' # 把字幕文件里的经纬度提取出来保存成numpy数据，这样方便读取节约时间
pkuBirdViewImg = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\pkuBirdView.png'

GNSS = np.load(posFile)
anchor_pos_list, anchor_neg_list = getAnchorPosNegIdx2(datasetPath, np.shape(GNSS)[0])
anchor_idx = [anchor_pos_list[i][0] for i in range(len(anchor_pos_list))]
fig, ax = plt.subplots()
plt.scatter(GNSS[:,1], GNSS[:,0], s=1, alpha=1) # 车辆行驶轨迹
plt.axis('equal')
plt.scatter(GNSS[anchor_idx,1], GNSS[anchor_idx,0], s = 10, c='k') # 所有锚点
for i in anchor_idx:
    pos_x = GNSS[i,1]
    pos_y = GNSS[i,0]
    ax.text(pos_x, pos_y, str(i), fontsize=10, alpha = 0.1)
plt.title('All anchors')

plt.show()