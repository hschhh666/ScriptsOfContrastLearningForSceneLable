import numpy as np
from cv2 import cv2
import re
import matplotlib.pyplot as plt
from utils import *
import random


featFile = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\deepLearningRes\\campusAllFrames\\memoryDic\\memoryNeg30FeatDim128e2e.npy' # 神经网络输出的数据集特征文件
subtitleFile = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\video_subtitle.srt' # 字幕文件，里面保存了每帧图片对应的经纬度
posFile = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\pos.npy' # 把字幕文件里的经纬度提取出来保存成numpy数据，这样方便读取节约时间
datasetPath = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\labeledData\\test2\\train' # 数据集路径
countVideoFrame = False
posNpyFile = True # 是从npy文件里读取每帧照片的位置，还是从字幕文件里读取位置


#======================加载数据，不涉及算法======================

# 加载特征文件
feat = np.load(featFile)#36924

# 加载锚点及其对应正负样本的id
anchor_pos_list, anchor_neg_list = getAnchorPosNegIdx(datasetPath)

# 加载视频文件，数一数视频一共有多少帧
if countVideoFrame:
    video = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\video.avi'
    video = cv2.VideoCapture(video)
    count = 0
    while 1:
        ret, img = video.read()
        if not ret:
            break
        count += 1
        if count %100 == 1:
            print('Reading video ',count)
    print('video total count = ',count)

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

#======================按距离间隔对帧进行重采样，这样相邻帧间的距离就是一致的了======================
interval = 5 # 每间隔interval米选择一帧
pickupFrame = np.zeros(np.shape(GNSS)[0], dtype = np.bool) # 这些帧中，相邻帧的间隔基本是一致的
pickupFrame[0] = 1
lastFram = 0
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

#======================画直方图可视化校验重采样后选择的相邻帧间的距离======================
gnss = GNSS[pickupFrame] # gnss里保存的就是每隔interval米选择的帧
delta = [np.sqrt(np.square(gnss[i][0]-gnss[i-1][0])+ np.square(gnss[i][1]-gnss[i-1][1])) for i in range(1, np.shape(gnss)[0])]
plt.figure()
plt.hist(delta, bins = 100) # 这里是为了做验证，看看选择出来的帧相邻是否是interval
plt.title('validate the interval between neighbor picked up frames, the interval should be %d m'%interval)

#======================找到所有拐弯的位置======================
turn_points = [] # 所有转弯的位置，格式为，[原始索引，重采样后索引，x，y]
for i, _ in enumerate(gnss):
    angleDelta = calculateAngel(gnss, i-2, i+2) # 根据yaw的变化来算出转弯
    if angleDelta >= 30:
        turn_points.append([resampledId_to_originId[i], i, gnss[i][0], gnss[i][1]])
print('turn points number is ', len(turn_points))


#======================计算锚点与正负样本间的平均距离======================
loop = 30
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


#======================取部分样本，检验锚点与其正负样本之间的相似度======================
for test_idx in range(1):
    test_idx = 1 # 检验锚点test_idx与其正负样本间的相似度
    anchor_num = len(anchor_pos_list)
    anchor_idx = anchor_pos_list[test_idx][0]
    pos_idx = anchor_pos_list[test_idx][1]
    neg_idx = anchor_pos_list[test_idx-1][1] + anchor_pos_list[(test_idx+1)%anchor_num][1]

    feat_anchor = feat[anchor_idx]
    feat_anchor = feat_anchor[:,np.newaxis]

    feat_pos = feat[anchor_pos_list[test_idx][1]]
    feat_neg = feat[anchor_neg_list[test_idx][1]]

    pos_avg_dis = np.mean(np.exp(np.matmul(feat_pos, feat_anchor)))
    neg_avg_dis = np.mean(np.exp(np.matmul(feat_neg, feat_anchor)))

    anchor_to_all_frames_dis = np.matmul(feat, feat_anchor)
    plt.figure()
    plt.title('anchor %d similarity to other frames\nto pos avg dis %f\nto neg avg dis %f'%(anchor_idx,pos_avg_dis, neg_avg_dis))
    plt.plot(anchor_to_all_frames_dis)

    plt.scatter(pos_idx,[anchor_to_all_frames_dis[i] for i in pos_idx],c='g') # 正样本为绿色点
    plt.scatter(neg_idx,[anchor_to_all_frames_dis[i] for i in neg_idx],c='b') # 负样本为蓝色点
    plt.scatter(anchor_idx,anchor_to_all_frames_dis[anchor_idx],c='r') # 锚点为红色点

#======================绘制锚点间的混淆矩阵======================
anchor_idx = [anchor_pos_list[i][0] for i in range(len(anchor_pos_list))]
anchors_feat = feat[anchor_idx]
anchors_feat_trans = anchors_feat.transpose(1,0)
dis_matrix = np.matmul(anchors_feat, anchors_feat_trans)
plt.figure()
plt.title('similarity between anchors')
plt.imshow(dis_matrix, interpolation='nearest', cmap=plt.cm.Greys)


#======================将视频重采样后，绘制帧间距离的混淆矩阵======================
resampled_feat = feat[pickupFrame] # pickupFrame数组中保存着重采样后帧的id，把这些帧选择出来
# resampled_feat = resampled_feat[0:1000] # 只选择一部分数据画出来，这是出于节省内存的考量
resampled_feat_trans = resampled_feat.transpose(1,0)
dis_matrix = np.matmul(resampled_feat,resampled_feat_trans)
plt.figure()
plt.title('resampled video. similarity between frames')
for i,_ in enumerate(turn_points):
    plt.scatter(turn_points[i][1], turn_points[i][1], c = 'r', s = 5) # 把计算出来的拐弯位置画在混淆矩阵上，来看看混淆矩阵突变的点是不是都是拐弯点

plt.imshow(dis_matrix, interpolation='nearest', cmap=plt.cm.Greys)

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
similarity_thresh = 0.2 # 当相邻帧的相似度低于该阈值时，则认为到了弯道
for i,_ in enumerate(neighbor_resampled_frame_similarity):
    if neighbor_resampled_frame_similarity[i] <= similarity_thresh:
        calculated_turn_points.append([resampledId_to_originId[i], i, gnss[i][0], gnss[i][1]])
        plt.scatter(i, neighbor_resampled_frame_similarity[i], c = 'g', marker='+',s=50) # 把根据阈值算出来的转弯点画在图上

# np.savetxt('D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\similarity_with_nextfram.txt', neighbor_resampled_frame_similarity,fmt='%.3f')

#======================将视频重采样后，绘制某个样本点与其他地点的相似度图======================
sampleId = 1799
plt.figure()
plt.scatter(sampleId,dis_matrix[sampleId][sampleId],c='r')
plt.plot(list(range(np.shape(dis_matrix)[0])),dis_matrix[sampleId])
plt.title('resampled video. sample %d'%sampleId)

#======================绘制某个点与所有视频帧的相似度======================
print('resampled id %d is origin id '%sampleId,end='')
sampleId = resampledId_to_originId[sampleId]
print(sampleId)
sample_feat = feat[sampleId]
sample_feat = sample_feat[:,np.newaxis]
dises = np.matmul(feat, sample_feat)
# dises = dises[pickupFrame]
plt.figure()
plt.plot(dises)
plt.title('origin video. sample %d'%sampleId)
plt.scatter(sampleId,dises[sampleId],c='r')

#======================把车辆行驶的轨迹画出来，标出拐弯处======================
plt.figure()
for i,_ in enumerate(turn_points):
    plt.scatter(turn_points[i][3], turn_points[i][2], c = 'r')
plt.plot(GNSS[:,1], GNSS[:,0])
plt.axis('equal')
plt.title('campus0107_trajectory')

#======================把计算得到的弯道画在车辆轨迹上======================
plt.figure()
for i,_ in enumerate(calculated_turn_points):
    plt.scatter(calculated_turn_points[i][3], calculated_turn_points[i][2], c = 'g')
plt.plot(GNSS[:,1], GNSS[:,0])
plt.axis('equal')
plt.title('campus0107_trajectory')





plt.show()
print('Program exit normally.')