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
import scipy.stats
import time
import random
from sklearn.metrics import confusion_matrix
import copy

gt_labelPath = '' # 默认没有锚点真值标签

featFile = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/deepLearningRes/labeledByTurn2/20210120_22_19_09_lossMethod_nce_NegNum_8_Model_alexnet_lr_0.03_decay_0.0001_bsz_8_featDim_128_contrasMethod_e2e_traditionalMethod_BestLoss_epoch60.npy'
datasetPath = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/RingroadTestData/labeledData/CampusSceneDataset_labeldByTurn2/anchorImgs'
gt_labelPath = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/RingroadTestData/labeledData/CampusSceneDataset_labeldByTurn2/label.txt'


# featFile = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/deepLearningRes/labeldByTurnAndDynamicTraffic/20210121_17_19_49_lossMethod_nce_NegNum_8_Model_alexnet_lr_0.03_decay_0.0001_bsz_8_featDim_128_contrasMethod_e2e_traditionalMethod_resume.npy'
featFile = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/deepLearningRes/labeldByTurnAndDynamicTraffic/20210121_17_27_12_lossMethod_nce_NegNum_8_Model_alexnet_lr_0.03_decay_0.0001_bsz_8_featDim_128_contrasMethod_e2e_traditionalMethod.npy'
# featFile = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/deepLearningRes/labeldByTurnAndDynamicTraffic_partSelected/20210130_19_27_44_lossMethod_nce_NegNum_8_Model_alexnet_lr_0.03_decay_0.0001_bsz_8_featDim_128_contrasMethod_e2e_traditionalMethod.npy'
datasetPath = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/RingroadTestData/labeledData/CampusSceneDataset_labeldByTurnAndDynamicTraffic/anchorImgs'
gt_labelPath = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/RingroadTestData/labeledData/CampusSceneDataset_labeldByTurnAndDynamicTraffic/label.txt'

# featFile = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/deepLearningRes/jiaoAnnotated/20210108_17_02_25_lossMethod_nce_NegNum_8_Model_alexnet_lr_0.03_decay_0.0001_bsz_8_featDim_128_contrasMethod_e2e_colorJitterAndRandomGray.npy'
# datasetPath = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/RingroadTestData/labeledData/campusSceneDataset_JiaoAnnotate1/anchorImgs'


subtitleFile = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\video_subtitle.srt' # 字幕文件，里面保存了每帧图片对应的经纬度
posFile = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\pos.npy' # 把字幕文件里的经纬度提取出来保存成numpy数据，这样方便读取节约时间

pkuBirdViewImg = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\pkuBirdView.png'
countVideoFrame = True
posNpyFile = True # 是从npy文件里读取每帧照片的位置，还是从字幕文件里读取位置


#======================加载数据，不涉及算法======================

# 加载特征文件
feat = np.load(featFile)#36924
frame_count = np.shape(feat)[0]

# 加载锚点及其对应正负样本的id
anchor_pos_list, anchor_neg_list = getAnchorPosNegIdx2(datasetPath, frame_count)
anchor_idx = [anchor_pos_list[i][0] for i in range(len(anchor_pos_list))]
anchors_feat = feat[anchor_idx]

# 加载视频文件，数一数视频一共有多少帧
if countVideoFrame:
    video = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\video.avi'
    video = cv2.VideoCapture(video)
    count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print('video total count = %d'%count)
    video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    


# 加载位置文件
if not posNpyFile:
    with open(subtitleFile) as f:
        lines = f.readlines()

    GNSS_x = []
    GNSS_y = []
    GNSS_yaw = []

    for i, line in enumerate(lines):
        match = re.match('GNSS_x:(.*)',line)
        if match != None:
            GNSS_x.append(float(match.group(1)))
        match = re.match('GNSS_y:(.*)',line)
        if match != None:
            GNSS_y.append(float(match.group(1)))
        match = re.match('IMU_yaw:(.*)',line)
        if match != None:
            GNSS_yaw.append(float(match.group(1))*180/np.pi)

    GNSS_x = np.array(GNSS_x) # x 南北方向，北为正方向
    GNSS_y = np.array(GNSS_y) # y 东西方向，东为正方向
    GNSS_yaw = np.array(GNSS_yaw)
    GNSS_x = GNSS_x[:,np.newaxis] # 扩充维度
    GNSS_y = GNSS_y[:,np.newaxis]
    GNSS_yaw = GNSS_yaw[:,np.newaxis] # yaw 的范围[0,360)，正北方向为0度，从0度向东转yaw增加
    GNSS = np.concatenate((GNSS_x,GNSS_y,GNSS_yaw),axis=1) # GNSS保存了每帧的位置坐标xy
    np.save(posFile, GNSS)

else:
    GNSS = np.load(posFile)

frameNum = np.shape(GNSS)[0] # 视频一共有多少帧

if gt_labelPath != '':
    f = open(gt_labelPath,'r')
    lines = f.readlines()
    gt_label_anchor = [int(i[0]) for i in lines]
    gt_label_allFrame = list(range(frame_count))
    for i, idx in enumerate(anchor_idx):
        if i == len(anchor_idx) - 1:
            break
        for j in range(anchor_idx[i], anchor_idx[i+1]):
            gt_label_allFrame[j] = gt_label_anchor[i]

    for i in range(anchor_idx[-1], frame_count):
        gt_label_allFrame[i] = gt_label_anchor[-1]
    for i in range(0, anchor_idx[0]):
        gt_label_allFrame[i] = gt_label_anchor[-1]

else:
    gt_label_anchor = [0 for i in anchor_idx]
    gt_label_allFrame = [0 for i in range(frame_count)]

gt_label_anchor = np.array(gt_label_anchor)
gt_label_allFrame = np.array(gt_label_allFrame)
# ======================t-SNE降维======================
cmap = 'rainbow'
lower_dim_anchor_feat = TSNE().fit_transform(anchors_feat)
fig = plt.figure()
fig.add_subplot(131)
plt.scatter(lower_dim_anchor_feat[:,0], lower_dim_anchor_feat[:,1], c=gt_label_anchor, cmap=cmap)
plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap))
for i in range(len(lower_dim_anchor_feat)):
    plt.annotate(anchor_idx[i], xy=(lower_dim_anchor_feat[i,0], lower_dim_anchor_feat[i,1]), xytext=(lower_dim_anchor_feat[i,0], lower_dim_anchor_feat[i,1]), alpha=0.1)
plt.title('anchor t-SNE')

# =======================PCA降维======================
pca = decomposition.PCA(n_components=2)
pca.fit(anchors_feat)
anchors_feat_pca = pca.fit_transform(anchors_feat)
fig.add_subplot(132)
plt.scatter(anchors_feat_pca[:,0], anchors_feat_pca[:,1], c = gt_label_anchor, cmap=cmap)
plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap))
plt.title('anchor PCA')


pca = decomposition.PCA(n_components=2)
pca.fit(feat)
feat_pca = pca.fit_transform(feat)
fig.add_subplot(133)
plt.scatter(feat_pca[:,0], feat_pca[:,1], c = gt_label_allFrame ,cmap = cmap, alpha=0.2)
plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap))
plt.title('allFrame PCA')

# plt.show()

# exit(0)

# ======================把特征画到视频上======================

# pca = decomposition.PCA(n_components=2)
# pca.fit(feat)
# lower_dim_feat = pca.fit_transform(feat)
# lower_dim_max = np.max(lower_dim_feat)
# lower_dim_max = max(-np.min(lower_dim_feat), lower_dim_max)
# lower_dim_max *= 1.2
# video.set(cv2.CAP_PROP_POS_FRAMES, 0)
# video_with_feat = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/campus_img_dataset/video_with_feat.avi'
# resize_ratio = 1
# width = int(video_width*resize_ratio)
# height = int(video_height*resize_ratio)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video_with_feat = cv2.VideoWriter(video_with_feat,fourcc,30.0,(width,height))
# while True:
#     fno = int(video.get(cv2.CAP_PROP_POS_FRAMES))
#     ret, frame = video.read()
#     if not ret:
#         break
#     frame = cv2.resize(frame,(int(video_width*resize_ratio),int(video_height*resize_ratio)))
#     feat_width = int(min(width, height) * 0.3)
#     pixel_size = feat_width / (lower_dim_max*2)
#     cv2.rectangle(frame, (width - feat_width, 0), (width, feat_width), (0,0,0), thickness=-1)
#     feat_x = lower_dim_feat[fno,0]
#     feat_y = lower_dim_feat[fno,1]
#     feat_x *= pixel_size
#     feat_y *= -pixel_size
#     feat_x = width - (feat_width/2) + feat_x
#     feat_y = (feat_width/2) + feat_y
#     feat_x = int(feat_x)
#     feat_y = int(feat_y)
#     cv2.circle(frame, (feat_x, feat_y), 2, (255,255,255), thickness=-1)
#     cv2.line(frame, (width - feat_width, int(feat_width/2)), (width, int(feat_width/2)), (255,255,255), thickness=1)
#     cv2.line(frame, (int(width - feat_width/2), 0), (int(width - feat_width/2), feat_width), (255,255,255), thickness=1)
#     video_with_feat.write(frame)
#     # cv2.imshow('frame', frame)
#     # cv2.waitKey(10)
#     if fno%100==0:
#         print(fno)


# exit(0)

#======================绘制某个点与所有视频帧的相似度======================
sampleId = 1300# 27755

# =====读取这帧视频=====
video.set(cv2.CAP_PROP_POS_FRAMES, sampleId)
ret, frame = video.read()
resize_ratio = 0.7
frame = cv2.resize(frame,(int(video_width*resize_ratio),int(video_height*resize_ratio)))
cv2.imshow('frame %d'%sampleId, frame)

# =====这帧与所有帧的相似度散点图=====
sample_feat = feat[sampleId]
sample_feat = sample_feat[:,np.newaxis]
dises = np.matmul(feat, sample_feat)

fig = plt.figure()
fig.add_subplot(121)
a = plt.scatter(list(range(len(dises))), dises[:,0], s = 1, zorder = 0)
plt.title('similarity of frame id %d to other frames'%sampleId)
plt.scatter(anchor_idx, dises[anchor_idx], c='k', s=15, zorder = 1)
plt.scatter(sampleId,dises[sampleId],c='r', zorder = 2)
plt.ylim((-1, 1.2))

# =====这帧与所有帧的折线图=====
fig.add_subplot(122)
a = plt.plot(dises, zorder = 0)
plt.title('similarity of frame id %d to other frames'%sampleId)
plt.scatter(anchor_idx, dises[anchor_idx], c='k', s=15, zorder = 1)
plt.scatter(sampleId,dises[sampleId],c='r', zorder = 2)
plt.ylim((-1, 1.2))

# =====把相似度画在轨迹上=====
# dises[0,0]=1
# dises[1,0]=-1
cmap = 'bwr'
fig, ax = plt.subplots()
plt.scatter(GNSS[:,1], GNSS[:,0], s=1, c=dises[:,0], cmap=cmap, zorder = 1) # 车辆行驶轨迹
plt.axis('equal')
plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap))
plt.scatter(GNSS[sampleId,1], GNSS[sampleId,0], s = 20, c='yellow') # 样本点
plt.scatter(GNSS[anchor_idx,1], GNSS[anchor_idx,0], s = 2, c='k') # 所有锚点
for i in anchor_idx:
    pos_x = GNSS[i,1]
    pos_y = GNSS[i,0]
    ax.text(pos_x, pos_y, str(i), fontsize=10, alpha = 0.1)
plt.title('frame %d similarity to other frames'%sampleId)

# =====把相似度画在轨迹上，投影到卫星图上=====
rotate = 0 # 角度制
shiftX = 399
shiftY = 650
dx = 0.775
dy = 0.775 # 以上参数都是手调的
alpha = 0.5

midX = np.average(GNSS[:,1])
midY = np.average(GNSS[:,0])
GNSS[:,1] -= midX
GNSS[:,0] -= midY
GNSS[:,0] = -GNSS[:,0]
rotate = rotate * np.pi / 180
x = np.cos(rotate)*GNSS[:,1] - np.sin(rotate) * GNSS[:,0]
y = np.sin(rotate)*GNSS[:,1] + np.cos(rotate) * GNSS[:,0]
GNSS[:,1] = x
GNSS[:,0] = y
GNSS[:,1] *= dx
GNSS[:,0] *= dy
GNSS[:,1] += shiftX
GNSS[:,0] += shiftY

cmap = 'bwr'
# # for sampleId in [26114, 32205, 1685, 31484, 20042, 19631,19285,2541,3375,4123,18768,17825,16421,14870,14795,15249,24878,27403,12639]:
# for sampleId in [1180,1917,2176,2434,3038, 3737,3971, 5962, 6378, 6751, 8241, 9768, 10779, 12088, 12265, 12550, 13031, 13961, 15159, 15652, 15866, 16088, 16727, 18402, 19204, 19788, 21080, 23229, 24028, 26257, 26566, 28633]:

sample_feat = feat[sampleId]
sample_feat = sample_feat[:,np.newaxis]
dises = np.matmul(feat, sample_feat)
# dises[0,0]=1
# dises[1,0] = -1
# fig = plt.figure(figsize=(20,20))
fig = plt.figure()
ax = fig.add_subplot(111)
img = gImage.imread(pkuBirdViewImg)
img = img[0:1087,:,:]
img[:,:,3] = alpha
ax.imshow(img, zorder = 0)
ax.scatter(GNSS[:,1], GNSS[:,0], s=1, c=dises[:,0], cmap=cmap, zorder = 1) # 车辆行驶轨迹
plt.axis('equal')
plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap))
plt.scatter(GNSS[sampleId,1], GNSS[sampleId,0], s = 20, c='yellow') # 样本点
plt.scatter(GNSS[anchor_idx,1], GNSS[anchor_idx,0], s = 2, c='k') # 所有锚点
for i in anchor_idx:
    pos_x = GNSS[i,1]
    pos_y = GNSS[i,0]
    ax.text(pos_x, pos_y, str(i), fontsize=10, alpha = 0.1)
plt.title('frame %d similarity to other frames'%sampleId)
    # plt.savefig(os.path.join('D:/Research/2020ContrastiveLearningForSceneLabel/Data/campus_img_dataset/res/jiaoAnnotated',str(sampleId)+'.png'))
    # plt.show()
    # plt.close('all')


# # ==========相邻帧的相似度画在轨迹上===========
# interver_frame = 10
# neighbor_sim = np.zeros(frame_count)
# for i in range(interver_frame - 1, frame_count):
#     tmp1 = feat[i]
#     tmp2 = feat[i-interver_frame]
#     fuck = tmp1*tmp2
#     neighbor_sim[i] = np.sum(tmp1*tmp2)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# img = gImage.imread(pkuBirdViewImg)
# img = img[0:1087,:,:]
# img[:,:,3] = alpha
# ax.imshow(img, zorder = 0)
# ax.scatter(GNSS[:,1], GNSS[:,0], s=1, c=neighbor_sim, cmap=cmap, zorder = 1) # 车辆行驶轨迹
# plt.axis('equal')
# plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap))
# plt.scatter(GNSS[anchor_idx,1], GNSS[anchor_idx,0], s = 2, c='k') # 所有锚点
# for i in anchor_idx:
#     pos_x = GNSS[i,1]
#     pos_y = GNSS[i,0]
#     ax.text(pos_x, pos_y, str(i), fontsize=10, alpha = 0.1)
# plt.title('neighbor %d frame similarity'%interver_frame)

# plt.figure()
# a = plt.plot(list(range(len(neighbor_sim))), neighbor_sim,  zorder = 0)
# plt.title('neighbor %d frame similarity'%interver_frame)
# plt.scatter(anchor_idx, neighbor_sim[anchor_idx], c='k', s=15, zorder = 1)

# =====这帧图像与所有图像相似度的直方图=====
fig = plt.figure()
fig.add_subplot(121)
dises = dises[:,0]
plt.hist(dises, bins=100,weights = np.zeros_like(dises) + 1 / len(dises))
plt.title('frame %d similarity to all frames histogram'%sampleId)
plt.ylim(0,0.08)
plt.xlim(-1,1)

# =====这帧图像与所有图像相似度的直方图=====
fig.add_subplot(122)
plt.hist(dises, bins=100)
plt.title('frame %d similarity to all frames histogram'%sampleId)
plt.ylim(0,1900)
plt.xlim(-1,1)

dises1 = dises[gt_label_allFrame != 2]
# =====这帧图像与所有非弯道相似度的直方图=====
fig = plt.figure()
fig.add_subplot(121)
plt.hist(dises1, bins=100,weights = np.zeros_like(dises1) + 1 / len(dises1))
plt.title('frame %d similarity to all not turn road histogram'%sampleId)
plt.ylim(0,0.08)
plt.xlim(-1,1)

# =====这帧图像与所有非弯道相似度的直方图=====
fig.add_subplot(122)
plt.hist(dises1, bins=100)
plt.title('frame %d similarity to all not turn road histogram'%sampleId)
plt.ylim(0,1900)
plt.xlim(-1,1)

dises2 = dises[gt_label_allFrame == 3]
# =====这帧图像与所有动态交通相似度的直方图=====
fig = plt.figure()
fig.add_subplot(121)
plt.hist(dises2, bins=100,weights = np.zeros_like(dises2) + 1 / len(dises2))
plt.title('frame %d similarity to all dynamic histogram'%sampleId)
plt.ylim(0,0.08)
plt.xlim(-1,1)

# =====这帧图像与所有动态交通相似度的直方图=====
fig.add_subplot(122)
plt.hist(dises2, bins=100)
plt.title('frame %d similarity to all dynamic histogram'%sampleId)
plt.ylim(0,1900)
plt.xlim(-1,1)


# # ======================可视化指定区域的特征======================
# roundLierIdx = list(range(0,1795)) + list(range(19802,19861)) + list(range(31697,34941))
# roundLierIdx = list(range(31705, 34168))
# dormIdx = list(range(24782,29554))
# natureIdx = list(range(6815, 7328)) + list(range(9043, 10009)) #+ list(range(13636, 13868)) + list(range(15209, 15316)) + list(range(18741, 18879))
# # roundLierIdx = list(range(0, 1729))
# # dormIdx = list(range(26329, 27700))
# # natureIdx = list(range(18733, 18871))

# candidateAreaIdx = roundLierIdx

# lierFeat = feat[candidateAreaIdx,:]
# dises = np.matmul(lierFeat, sample_feat)
# dises = dises[:,0]

# dises[0] = 1
# dises[1] = -1

# # =====这帧图像与某区域图像相似度的直方图=====
# plt.figure()
# plt.hist(dises, bins=100,weights = np.zeros_like(dises) + 1 / len(dises))
# # plt.hist(dises, bins = 100)
# plt.ylim(0,0.08)
# plt.xlim(-1,1)
# plt.title('frame %d similarity to area frames histogram'%sampleId)


# # =====这帧图像与某区域图像相似度的离散成n类画在卫星图上=====
# fig = plt.figure()
# ax = fig.add_subplot(111)
# img = gImage.imread(pkuBirdViewImg)
# img = img[0:1087,:,:]
# img[:,:,3] = alpha
# ax.imshow(img, zorder = 0)

# idx = np.array(candidateAreaIdx)[dises > 0.7]
# ax.scatter(GNSS[idx,1], GNSS[idx,0], s=1, c='r') # 车辆行驶轨迹

# idx = np.array(candidateAreaIdx)[dises < 0.4]
# ax.scatter(GNSS[idx,1], GNSS[idx,0], s=1, c='b') # 车辆行驶轨迹

# idx = np.array(candidateAreaIdx)[np.logical_and(dises > 0.4 , dises < 0.7)]
# ax.scatter(GNSS[idx,1], GNSS[idx,0], s=1, c='g') # 车辆行驶轨迹


# plt.axis('equal')
# plt.scatter(GNSS[sampleId,1], GNSS[sampleId,0], s = 20, c='yellow') # 样本点
# plt.scatter(GNSS[anchor_idx,1], GNSS[anchor_idx,0], s = 2, c='k') # 所有锚点
# for i in anchor_idx:
#     pos_x = GNSS[i,1]
#     pos_y = GNSS[i,0]
#     ax.text(pos_x, pos_y, str(i), fontsize=10, alpha = 0.1)
# plt.title('frame %d similarity to area frames--discretization'%sampleId)


# # =====这帧图像与某区域图像相似度画在卫星图上=====
# fig = plt.figure()
# ax = fig.add_subplot(111)
# img = gImage.imread(pkuBirdViewImg)
# img = img[0:1087,:,:]
# img[:,:,3] = alpha
# ax.imshow(img, zorder = 0)
# ax.scatter(GNSS[candidateAreaIdx,1], GNSS[candidateAreaIdx,0], s=1, c=dises, cmap=cmap, zorder = 1) # 车辆行驶轨迹
# plt.axis('equal')
# plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap))
# plt.scatter(GNSS[sampleId,1], GNSS[sampleId,0], s = 20, c='yellow') # 样本点
# plt.scatter(GNSS[anchor_idx,1], GNSS[anchor_idx,0], s = 2, c='k') # 所有锚点
# for i in anchor_idx:
#     pos_x = GNSS[i,1]
#     pos_y = GNSS[i,0]
#     ax.text(pos_x, pos_y, str(i), fontsize=10, alpha = 0.1)
# plt.title('frame %d similarity to area frames'%sampleId)

# # =====把区域的轨迹画在卫星图上=====
# fig = plt.figure()
# ax = fig.add_subplot(111)
# img = gImage.imread(pkuBirdViewImg)
# img = img[0:1087,:,:]
# img[:,:,3] = alpha
# ax.imshow(img, zorder = 0)
# ax.scatter(GNSS[roundLierIdx,1], GNSS[roundLierIdx,0], s=1, c='b')
# ax.scatter(GNSS[dormIdx,1], GNSS[dormIdx,0], s=1, c='r')
# ax.scatter(GNSS[natureIdx,1], GNSS[natureIdx,0], s=1, c='g')
# plt.axis('equal')
# plt.title('area traj')

# # =====计算区域内、区域间的相似度=====
# tmpFeat = feat[roundLierIdx+dormIdx+natureIdx, :]

# roundLierFeat = tmpFeat[0:len(roundLierIdx),:]
# dormFeat = tmpFeat[len(roundLierIdx):(len(roundLierIdx + dormIdx)),:]
# natureFeat = tmpFeat[len(roundLierIdx + dormIdx):len(roundLierIdx + dormIdx+natureIdx),:]

# roudLierFeat_trans = roundLierFeat.transpose(1,0)
# dormFeat_trans = dormFeat.transpose(1,0)
# natureFeat_trans = natureFeat.transpose(1,0)

# roundLier_matrix = np.matmul(roundLierFeat, roudLierFeat_trans)
# dorm_matrix = np.matmul(dormFeat, dormFeat_trans)
# nature_matrix = np.matmul(natureFeat, natureFeat_trans)

# sr = np.shape(roundLier_matrix)[0]
# roundLier_sim = (np.sum(roundLier_matrix) - roundLier_matrix.trace()) / (sr*(sr-1))
# sd = np.shape(dorm_matrix)[0]
# dorm_sim = (np.sum(dorm_matrix) - dorm_matrix.trace()) / (sd*(sd-1))
# sn = np.shape(nature_matrix)[0]
# nature_sim = (np.sum(nature_matrix) - nature_matrix.trace()) / (sn*(sn-1))

# r_d = np.average(np.matmul(roundLierFeat, dormFeat_trans))
# r_n = np.average(np.matmul(roundLierFeat, natureFeat_trans))
# d_n = np.average(np.matmul(dormFeat, natureFeat_trans))

# print('inside area similarity', roundLier_sim, nature_sim, dorm_sim)
# print('betewwn area similarity', r_n, r_d, d_n)

# # =====对区域进行聚类=====
# cluster_num = 2
# cmap = 'rainbow'
# kmeans = KMeans(n_clusters=cluster_num).fit(roundLierFeat)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# img = gImage.imread(pkuBirdViewImg)
# img = img[0:1087,:,:]
# img[:,:,3] = alpha
# ax.imshow(img, zorder = 0)
# ax.scatter(GNSS[roundLierIdx,1], GNSS[roundLierIdx,0], s=1, c=kmeans.labels_, cmap=cmap, zorder = 1) # 车辆行驶轨迹
# plt.axis('equal')
# plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap))
# plt.scatter(GNSS[anchor_idx,1], GNSS[anchor_idx,0], s = 2, c='k') # 所有锚点
# plt.title('area cluster--%d'%cluster_num)

# # ======计算直路、弯路内部相似度与之间的相似度=====
# straight_feat = feat[gt_label_allFrame == 1]
# straight_feat_trans = straight_feat.transpose(1,0)
# turn_feat = feat[gt_label_allFrame == 2]
# turn_feat_trans = turn_feat.transpose(1,0)
# dyna_feat = feat[gt_label_allFrame == 3]
# dyna_feat_trans = dyna_feat.transpose(1,0)

# straight_sim = np.matmul(straight_feat, straight_feat_trans)
# eye_matrix = np.logical_not(np.eye(np.shape(straight_sim)[0], dtype=bool))
# straight_sim = straight_sim[eye_matrix]
# straight_sim_avg = np.average(straight_sim)
# straight_sim_std = np.std(straight_sim)


# turn_sim = np.matmul(turn_feat, turn_feat_trans)
# eye_matrix = np.logical_not(np.eye(np.shape(turn_sim)[0], dtype=bool))
# turn_sim = turn_sim[eye_matrix]
# turn_sim_avg = np.average(turn_sim)
# turn_sim_std = np.std(turn_sim)

# dyna_sim = np.matmul(dyna_feat, dyna_feat_trans)
# eye_matrix = np.logical_not(np.eye(np.shape(dyna_sim)[0], dtype=bool))
# dyna_sim = dyna_sim[eye_matrix]
# dyna_sim_avg = np.average(dyna_sim)
# dyna_sim_std = np.std(dyna_sim)

# straight_turn_sim = np.average(np.matmul(straight_feat, turn_feat_trans))
# straight_dyna_sim = np.average(np.matmul(straight_feat, dyna_feat_trans))
# turn_dyna_sim = np.average(np.matmul(turn_feat, dyna_feat_trans))

# straight_turn_std = np.std(np.matmul(straight_feat, turn_feat_trans))
# straight_dyna_std = np.std(np.matmul(straight_feat, dyna_feat_trans))
# turn_dyna_std = np.std(np.matmul(turn_feat, dyna_feat_trans))

# print('straight_sim_avg %f\n turn_sim_avg %f\n dyna_sim_avg %f\n straight_turn_sim %f\n straight_dyna_sim %f\n turn_dyna_sim %f\n'%(straight_sim_avg, turn_sim_avg, dyna_sim_avg,straight_turn_sim, straight_dyna_sim, turn_dyna_sim))
# print('straight_sim_std %f\n turn_sim_std %f\n dyna_sim_std %f \n straight_turn_std %f\n straight_dyna_std %f \n turn_dyna_std %f\n'%(straight_sim_std, turn_sim_std, dyna_sim_std, straight_turn_std, straight_dyna_std, turn_dyna_std))

# ========计算每个样本与所有视频帧的相似度，统计相似度分布，作为该样本的新特征=========
testing_frome_frame_id = 32701

# all_feat_sim = np.matmul(feat, feat.transpose(1,0))
# sim_hist_feat = []
# bins = 100
# for i in range(frame_count):
#     cur_sim = all_feat_sim[i]
#     mask = np.ones(frame_count, dtype=bool)
#     mask[i] = False
#     mask[testing_frome_frame_id:] = False
#     cur_sim = cur_sim[mask]
#     cur_hist = np.histogram(cur_sim, bins=bins, weights = np.zeros_like(cur_sim) + 1 / len(cur_sim) ,range=(-1,1))[0]
#     sim_hist_feat.append(cur_hist)

# sim_hist_feat = np.array(sim_hist_feat)
# sim_hist_feat_path = featFile[0:-4] + '_simHistFeat.npy'
# np.save(sim_hist_feat_path, sim_hist_feat)
sim_hist_feat_path = featFile[0:-4] + '_simHistFeat.npy'
sim_hist_feat = np.load(sim_hist_feat_path)


# straight_sim = all_feat_sim[gt_label_allFrame==1]
# turn_sim = all_feat_sim[gt_label_allFrame==2]
# dyna_sim = all_feat_sim[gt_label_allFrame==3]

# straight_sim = straight_sim.reshape(-1)
# turn_sim = turn_sim.reshape(-1)
# dyna_sim = dyna_sim.reshape(-1)

# plt.figure()
# plt.hist(straight_sim, bins=100,weights = np.zeros_like(straight_sim) + 1 / len(straight_sim))
# plt.title('straight avg hist')
# plt.ylim(0,0.03)
# plt.xlim(-1,1)

# plt.figure()
# plt.hist(turn_sim, bins=100,weights = np.zeros_like(turn_sim) + 1 / len(turn_sim))
# plt.title('turn avg hist')
# plt.ylim(0,0.03)
# plt.xlim(-1,1)

# plt.figure()
# plt.hist(dyna_sim, bins=100,weights = np.zeros_like(dyna_sim) + 1 / len(dyna_sim))
# plt.title('dyna avg hist')
# plt.ylim(0,0.03)
# plt.xlim(-1,1)


# # ============根据相似度进行分类============
# straight_sim = feat[gt_label_allFrame==1]
# turn_sim = feat[gt_label_allFrame==2]
# dyna_sim = feat[gt_label_allFrame==3]

# straight_sim = np.average(straight_sim, axis=0)
# turn_sim = np.average(turn_sim, axis=0)
# dyna_sim = np.average(dyna_sim, axis=0)

# cal_label_allFrame = [3 for i in range(frame_count)]
# for i in range(frame_count):
#     cur_feat = feat[i]
#     js_s = np.sum(cur_feat * straight_sim)
#     js_t = np.sum(cur_feat * turn_sim)
#     js_d = np.sum(cur_feat * dyna_sim)
#     if js_s > js_t and js_s > js_d:
#         cal_label_allFrame[i] = 1
#     elif js_t > js_s and js_t > js_d:
#         cal_label_allFrame[i] = 2

# classes = ['straight','turn','dyna']
# cm = confusion_matrix(gt_label_allFrame, cal_label_allFrame)
# plot_confusion_matrix(cm, classes, title='sim confusion matrix')


# =================根据全局相似度分布进行分类，真值=======================
train_gt_label_allFrame = gt_label_allFrame.copy()
train_gt_label_allFrame[testing_frome_frame_id:] = 0

straight_sim_hist_feat = sim_hist_feat[train_gt_label_allFrame==1]
turn_sim_hist_feat = sim_hist_feat[train_gt_label_allFrame==2]
dyna_sim_hist_feat = sim_hist_feat[train_gt_label_allFrame==3]

straight_sim_hist_feat = np.average(straight_sim_hist_feat, axis=0)
turn_sim_hist_feat = np.average(turn_sim_hist_feat, axis=0)
dyna_sim_hist_feat = np.average(dyna_sim_hist_feat, axis=0)

cal_label_allFrame = [3 for i in range(frame_count)]
for i in range(frame_count):
    cur_hist = sim_hist_feat[i]
    js_s = JS_D(cur_hist, straight_sim_hist_feat)
    js_t = JS_D(cur_hist, turn_sim_hist_feat)
    js_d = JS_D(cur_hist, dyna_sim_hist_feat)
    if js_s < js_t and js_s < js_d:
        cal_label_allFrame[i] = 1
    elif js_t < js_s and js_t < js_d:
        cal_label_allFrame[i] = 2
    
classes = ['straight','turn','dyna']
cm = confusion_matrix(gt_label_allFrame[0:testing_frome_frame_id], cal_label_allFrame[0:testing_frome_frame_id])
plot_confusion_matrix(cm, classes, title='GT hist confusion matrix on training set')



# ==================画全局相似度分布========================
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
plt.bar(range(100), straight_sim_hist_feat,width=1, align='edge')
plt.ylim(0,0.04)
ax.set_xticks([0,25,50,75,99])
ax.set_xticklabels(['-1','-0.5','0','0.5','1'])
plt.title('GT straight scene descriptor')


ax = fig.add_subplot(1, 3, 2)
plt.bar(range(100), turn_sim_hist_feat,width=1, align='edge')
plt.ylim(0,0.04)
ax.set_xticks([0,25,50,75,99])
ax.set_xticklabels(['-1','-0.5','0','0.5','1'])
plt.title('GT turn scene descriptor')


ax = fig.add_subplot(1, 3, 3)
plt.bar(range(100), dyna_sim_hist_feat,width=1, align='edge')
plt.ylim(0,0.04)
ax.set_xticks([0,25,50,75,99])
ax.set_xticklabels(['-1','-0.5','0','0.5','1'])
plt.title('GT dyna scene descriptor')


# ===================计算scene descriptor========================
straight_seed_hist = sim_hist_feat[5589]
turn_seed_hist = sim_hist_feat[24708]
dyna_seed_hist = sim_hist_feat[27768]

straight_scene_descriptor = []
turn_scene_descriptor = []
dyna_scene_descriptor = []

for i in range(testing_frome_frame_id):
    cur_hist = sim_hist_feat[i]
    js_s = JS_D(cur_hist, straight_seed_hist)
    js_t = JS_D(cur_hist, turn_seed_hist)
    js_d = JS_D(cur_hist, dyna_seed_hist)
    if js_s < js_t and js_s < js_d:
        straight_scene_descriptor.append(cur_hist)
    elif js_t < js_s and js_t < js_d:
        turn_scene_descriptor.append(cur_hist)
    else:
        dyna_scene_descriptor.append(cur_hist)

straight_scene_descriptor = np.average(np.array(straight_scene_descriptor), axis=0)
turn_scene_descriptor =  np.average(np.array(turn_scene_descriptor), axis=0)
dyna_scene_descriptor =  np.average(np.array(dyna_scene_descriptor), axis=0)

# =================画scene descriptor=================
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
plt.bar(range(100), straight_scene_descriptor,width=1, align='edge')
# plt.plot(straight_scene_descriptor)
plt.ylim(0,0.04)
ax.set_xticks([0,25,50,75,99])
ax.set_xticklabels(['-1','-0.5','0','0.5','1'])
plt.title('cal straight scene descriptor')

ax = fig.add_subplot(1, 3, 2)
plt.bar(range(100), turn_scene_descriptor,width=1, align='edge')
# plt.plot(turn_scene_descriptor)
plt.ylim(0,0.04)
ax.set_xticks([0,25,50,75,99])
ax.set_xticklabels(['-1','-0.5','0','0.5','1'])
plt.title('cal turn scene descriptor')


ax = fig.add_subplot(1, 3, 3)
plt.bar(range(100), dyna_scene_descriptor,width=1, align='edge')
# plt.plot(dyna_scene_descriptor)
plt.ylim(0,0.04)
ax.set_xticks([0,25,50,75,99])
ax.set_xticklabels(['-1','-0.5','0','0.5','1'])
plt.title('cal dyna scene descriptor')

# ==================根据scene descriptor进行场景分类====================
cal_label_allFrame = [3 for i in range(frame_count)]
for i in range(frame_count):
    cur_hist = sim_hist_feat[i]
    js_s = JS_D(cur_hist, straight_scene_descriptor)
    js_t = JS_D(cur_hist, turn_scene_descriptor)
    js_d = JS_D(cur_hist, dyna_scene_descriptor)
    if js_s < js_t and js_s < js_d:
        cal_label_allFrame[i] = 1
    elif js_t < js_s and js_t < js_d:
        cal_label_allFrame[i] = 2

cal_label_allFrame_path = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/campus_img_dataset/cal_label_allFrame.txt'
f = open(cal_label_allFrame_path,'w')
for i in cal_label_allFrame:
    f.write(str(i)+'\n')
f.close()
    
classes = ['straight','turn','dyna']
cm = confusion_matrix(gt_label_allFrame[:testing_frome_frame_id], cal_label_allFrame[:testing_frome_frame_id])
plot_confusion_matrix(cm, classes, title='cal hist confusion matrix on training set')

print('<================Training set classification report================>')
print(classification_report(gt_label_allFrame[:testing_frome_frame_id],cal_label_allFrame[:testing_frome_frame_id],digits=3))
print('<================Training set classification report================>')

# ===================将分类结果投影到轨迹上=======================
cmap = 'rainbow'
fig = plt.figure()
ax = fig.add_subplot(111)
img = gImage.imread(pkuBirdViewImg)
img = img[0:1087,:,:]
img[:,:,3] = alpha
ax.imshow(img, zorder = 0)
ax.scatter(GNSS[0:testing_frome_frame_id,1], GNSS[0:testing_frome_frame_id,0], s=1, c=cal_label_allFrame[0:testing_frome_frame_id], cmap=cmap, zorder = 1) # 车辆行驶轨迹
plt.axis('equal')
plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap))

train_anchor_idx = np.array(anchor_idx)
train_anchor_idx = train_anchor_idx[train_anchor_idx < testing_frome_frame_id]

plt.scatter(GNSS[train_anchor_idx,1], GNSS[train_anchor_idx,0], s = 2, c='k') # 所有锚点
for i in train_anchor_idx:
    pos_x = GNSS[i,1]
    pos_y = GNSS[i,0]
    ax.text(pos_x, pos_y, str(i), fontsize=10, alpha = 0.1)
plt.title('scene classificaton on training set -- base on cal scene descriptor')


# straight_sim_hist_feat = list(sim_hist_feat[gt_label_allFrame==1])
# turn_sim_hist_feat = list(sim_hist_feat[gt_label_allFrame==2])
# dyna_sim_hist_feat = list(sim_hist_feat[gt_label_allFrame==3])

# # ======计算特征相似度分布间的JS散度==========


# loop = 200000
# res = 0
# start = time.time()
# for i in range(loop):
#     p = random.sample(dyna_sim_hist_feat, 1)[0]
#     q = random.sample(turn_sim_hist_feat, 1)[0]
#     res += js(p,q)
# res /= loop
# print('time %f\n'%(time.time()-start))
# print(res)

print('<================Testing set classification report================>')
print(classification_report(gt_label_allFrame[testing_frome_frame_id:],cal_label_allFrame[testing_frome_frame_id:],digits=3))
print('<================Testing set classification report================>')
classes = ['straight','turn','dyna']
cm = confusion_matrix(gt_label_allFrame[testing_frome_frame_id:], cal_label_allFrame[testing_frome_frame_id:])
plot_confusion_matrix(cm, classes, title='cal hist confusion matrix on testing set')


cmap = 'rainbow'
fig = plt.figure()
ax = fig.add_subplot(111)
img = gImage.imread(pkuBirdViewImg)
img = img[0:1087,:,:]
img[:,:,3] = alpha
ax.imshow(img, zorder = 0)
ax.scatter(GNSS[testing_frome_frame_id:,1], GNSS[testing_frome_frame_id:,0], s=1, c=cal_label_allFrame[testing_frome_frame_id:], cmap=cmap, zorder = 1) # 车辆行驶轨迹
plt.axis('equal')
plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap))

test_anchor_idx = np.array(anchor_idx)
test_anchor_idx = test_anchor_idx[test_anchor_idx >= testing_frome_frame_id]

plt.scatter(GNSS[test_anchor_idx,1], GNSS[test_anchor_idx,0], s = 2, c='k') # 所有锚点
for i in test_anchor_idx:
    pos_x = GNSS[i,1]
    pos_y = GNSS[i,0]
    ax.text(pos_x, pos_y, str(i), fontsize=10, alpha = 0.1)
plt.title('scene classificaton on testing set-- base on cal scene descriptor')

testID = 35898
test = sim_hist_feat[testID]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.bar(range(100), test,width=1, align='edge')
plt.ylim(0,0.04)
ax.set_xticks([0,25,50,75,99])
ax.set_xticklabels(['-1','-0.5','0','0.5','1'])
plt.title('%d hist'%testID)



#======================计算锚点与正负样本间的平均距离======================
loop = 1
avg_pos_dis = 0
avg_neg_dis = 0

for i in range(loop):
    imgs_idx = []
    for i in anchor_pos_list:
        imgs_idx += i[1]
    imgs_feat = feat[imgs_idx] # 获取所有样本点的特征

    pos_idx = [random.sample(anchor_pos_list[i][1],1)[0] for i in range(len(anchor_pos_list)) for j in anchor_pos_list[i][1]]
    pos_feat = feat[pos_idx]
    pos_feat = pos_feat.transpose(1,0)
    matrix = np.matmul(imgs_feat, pos_feat)
    matrix = np.exp(matrix)
    pos_dis = matrix.trace()/len(imgs_idx)
    avg_pos_dis += pos_dis

    neg_idx = [random.sample(anchor_neg_list[i][1],1)[0] for i in range(len(anchor_neg_list)) for j in anchor_pos_list[i][1]]
    neg_feat = feat[neg_idx]
    neg_feat = neg_feat.transpose(1,0)
    matrix = np.matmul(imgs_feat, neg_feat)
    matrix = np.exp(matrix)
    neg_dis = matrix.trace()/len(imgs_idx)
    avg_neg_dis += neg_dis
print('anchor to positive average distance is %f, to negative is %f'%(avg_pos_dis/loop, avg_neg_dis/loop))

# # ====================== 全局聚类 ======================
# cluster_num = 3
# cmap = 'rainbow'
# kmeans = KMeans(n_clusters=cluster_num).fit(feat)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# img = gImage.imread(pkuBirdViewImg)
# img = img[0:1087,:,:]
# img[:,:,3] = alpha
# ax.imshow(img, zorder = 0)
# ax.scatter(GNSS[:,1], GNSS[:,0], s=1, c=kmeans.labels_, cmap=cmap, zorder = 1) # 车辆行驶轨迹
# plt.axis('equal')
# plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap))
# plt.scatter(GNSS[anchor_idx,1], GNSS[anchor_idx,0], s = 2, c='k') # 所有锚点
# plt.title('cluster--%d'%cluster_num)




































plt.show()
exit(0) 
# =======================================================END=============================================================================
# ====================== 计算相邻帧间的距离 ======================
neighborFrameDis = [np.sqrt(np.square(GNSS[i][0]-GNSS[i-1][0]) + np.square(GNSS[i][1]-GNSS[i-1][1])) for i in range(1,np.shape(GNSS)[0])]
plt.figure()
plt.plot(neighborFrameDis)
plt.title('distance (meter) between neighbor frame')

#======================按距离间隔对帧进行重采样，这样相邻帧间的距离就是一致的了======================
interval = 1 # 每间隔interval米选择一帧，即对原始视频帧进行最近邻降采样
pickupFrame = np.zeros(np.shape(GNSS)[0], dtype = np.bool) # bool型数组，大小就是视频帧总数。pickupFrame[i]表示是否采样第i帧，True则采样。
pickupFrame[0] = True
lastFram = 0 # 表示上一个采样到的帧的id，从第0帧开始采样
resampledId_to_originId = [0] # 重采样后的某一帧对应的原始帧的索引
for i, _ in enumerate(GNSS):
    deltaDis1 = np.sqrt(np.square(GNSS[i][0]-GNSS[lastFram][0])+ np.square(GNSS[i][1]-GNSS[lastFram][1]))
    if i != frameNum - 1:
        deltaDis2 = np.sqrt(np.square(GNSS[i+1][0]-GNSS[lastFram][0])+ np.square(GNSS[i+1][1]-GNSS[lastFram][1]))
        if deltaDis1 <= interval and deltaDis2 >= interval:
            if abs(deltaDis1 - interval) < abs(deltaDis2 - interval):
                pickupFrame[i] = 1
                lastFram = i
                resampledId_to_originId.append(i)
            else:
                pickupFrame[i+1] = 1
                lastFram = i+1
                resampledId_to_originId.append(i+1)
        if deltaDis1 > interval:
            print('[Warning] Downsample threshold distance %d mis less than vehicle move distance!!! Frame id = %d'%(interval, i))
        if deltaDis1 > interval:
            pickupFrame[i] = 1
            lastFram = i
            resampledId_to_originId.append(i)

#======================画直方图可视化校验重采样后选择的相邻帧间的距离======================
gnss = GNSS[pickupFrame] # gnss里保存的就是每隔interval米选择的帧，即重采样后的帧的GNSS信息
print('After %dm downsample, frames number changes from %d to %d'%(interval ,np.shape(GNSS)[0], np.shape(gnss)[0]))
delta = [np.sqrt(np.square(gnss[i][0]-gnss[i-1][0])+ np.square(gnss[i][1]-gnss[i-1][1])) for i in range(1, np.shape(gnss)[0])]
plt.figure()
plt.hist(delta, bins = 100) # 这里是为了做验证，看看选择出来的帧相邻是否是interval
plt.title('validate the interval between downsampled frames, the interval should be %d m'%interval)

#======================找到所有拐弯的位置======================
turn_points = [] # 所有转弯的位置，格式为，[原始索引，重采样后索引，x，y]
for i, _ in enumerate(gnss):
    angleDelta = calculateAngel(gnss, i-1, i+1) # 根据yaw的变化来算出转弯
    if angleDelta >= 30:
        turn_points.append([resampledId_to_originId[i], i, gnss[i][0], gnss[i][1]])
print('calculated from GNSS file, turn points number is ', len(turn_points))

#======================取部分样本，检验锚点与其正负样本之间的相似度======================
for test_idx in range(1):
    test_idx = 10 # 检验锚点test_idx与其正负样本间的相似度
    anchor_num = len(anchor_pos_list)
    test_anchor_idx = anchor_pos_list[test_idx][0]
    pos_idx = anchor_pos_list[test_idx][1]
    neg_idx = anchor_pos_list[test_idx-1][1] + anchor_pos_list[(test_idx+1)%anchor_num][1]

    feat_anchor = feat[test_anchor_idx]
    feat_anchor = feat_anchor[:,np.newaxis]

    feat_pos = feat[anchor_pos_list[test_idx][1]]
    feat_neg = feat[anchor_neg_list[test_idx][1]]

    pos_avg_dis = np.mean(np.exp(np.matmul(feat_pos, feat_anchor)))
    neg_avg_dis = np.mean(np.exp(np.matmul(feat_neg, feat_anchor)))

    anchor_to_all_frames_dis = np.matmul(feat, feat_anchor)
    plt.figure()
    plt.title('anchor fno %d similarity to other frames\nto pos avg dis %f\nto neg avg dis %f'%(test_anchor_idx,pos_avg_dis, neg_avg_dis))
    plt.plot(anchor_to_all_frames_dis)

    plt.scatter(pos_idx,[anchor_to_all_frames_dis[i] for i in pos_idx],c='g') # 正样本为绿色点
    plt.scatter(neg_idx,[anchor_to_all_frames_dis[i] for i in neg_idx],c='b') # 负样本为蓝色点
    plt.scatter(test_anchor_idx,anchor_to_all_frames_dis[test_anchor_idx],c='r') # 锚点为红色点
    plt.scatter(test_anchor_idx, anchor_to_all_frames_dis[test_anchor_idx],  marker='+', c = 'r') # 红色十字为所有锚点

#======================绘制锚点间的混淆矩阵======================
anchors_feat_trans = anchors_feat.transpose(1,0)
dis_matrix = np.matmul(anchors_feat, anchors_feat_trans)
plt.figure()
plt.title('similarity between anchors')
plt.imshow(dis_matrix, interpolation='nearest', cmap=cmap)


#======================将视频重采样后，绘制帧间距离的混淆矩阵======================
resampled_feat = feat[pickupFrame] # pickupFrame数组中保存着重采样后帧的id，把这些帧选择出来
# resampled_feat = feat
resampled_feat_trans = resampled_feat.transpose(1,0)
dis_matrix = np.matmul(resampled_feat,resampled_feat_trans)
plt.figure()
plt.title('resampled video. similarity between frames')
# for i,_ in enumerate(turn_points):
#     plt.scatter(turn_points[i][1], turn_points[i][1], c = 'r', s = 5) # 把计算出来的拐弯位置画在混淆矩阵上，来看看混淆矩阵突变的点是不是都是拐弯点
# plt.scatter(anchor_idx, anchor_idx, c = 'k', marker= '+', s=1)
plt.imshow(dis_matrix, interpolation='nearest', cmap=cmap)
resampled_feat = feat[pickupFrame]
#======================将视频重采样后，绘制相邻帧间的相似度曲线======================
resampled_num = len(resampled_feat)
neighbor_resampled_frame_similarity = [np.sum(resampled_feat[i]*resampled_feat[i+1]) for i in range(resampled_num-1)]
plt.figure()
plt.plot(neighbor_resampled_frame_similarity) # 把相邻帧相似度画成曲线
plt.title('resampled video, neighbor frame similarity')
for i,_ in enumerate(turn_points):
    plt.scatter(turn_points[i][1], neighbor_resampled_frame_similarity[turn_points[i][1]], c = 'r',s=25) # 把根据yaw计算出来的转弯点画在图上

#======================根据相邻帧的相似度计算拐弯位置======================
calculated_turn_points = [] # 根据相邻帧的相似度计算出来的拐弯位置，数据格式[原始索引，重采样后索引，x，y]
similarity_thresh = 0.2
 # 当相邻帧的相似度低于该阈值时，则认为到了弯道
for i,_ in enumerate(neighbor_resampled_frame_similarity):
    if neighbor_resampled_frame_similarity[i] <= similarity_thresh:
        calculated_turn_points.append([resampledId_to_originId[i], i, gnss[i][0], gnss[i][1]])
        plt.scatter(i, neighbor_resampled_frame_similarity[i], c = 'g', marker='+',s=50) # 把根据阈值算出来的转弯点画在图上

# np.savetxt('D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\similarity_with_nextfram.txt', neighbor_resampled_frame_similarity,fmt='%.3f')

# #======================将视频重采样后，绘制某个样本点与其他地点的相似度图======================
# sampleId = 1000
# plt.figure()
# plt.scatter(sampleId,dis_matrix[sampleId][sampleId],c='r')
# plt.plot(list(range(np.shape(dis_matrix)[0])),dis_matrix[sampleId])
# plt.title('similarity of resampled frame id %d to other resampled frames'%sampleId)

# #======================把车辆行驶的轨迹画出来，标出拐弯处======================
# plt.figure()
# for i,_ in enumerate(turn_points):
#     plt.scatter(turn_points[i][3], turn_points[i][2], c = 'r')
# plt.plot(GNSS[:,1], GNSS[:,0])
# plt.axis('equal')
# plt.title('campus0107_trajectory, ground truth turn')

#======================把计算得到的弯道画在车辆轨迹上======================
# plt.figure()
# for i,_ in enumerate(calculated_turn_points):
#     plt.scatter(calculated_turn_points[i][3], calculated_turn_points[i][2], c = 'g')
# plt.plot(GNSS[:,1], GNSS[:,0])
# plt.axis('equal')
# plt.title('campus0107_trajectory, calculated from features turn')





plt.show()
print('Program exit normally.')