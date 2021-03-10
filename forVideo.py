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

pkuBirdViewImg = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\pkuBirdView_gray.png'
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


# img = cv2.imread(pkuBirdViewImg)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
# cv2.imwrite('D:/Research/2020ContrastiveLearningForSceneLabel/Data/pkuBirdView_gray.png', img)
# exit(0)


start = 5530
end = 6011


#######################数据加载完毕#####################

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# ratio = 1
# out = cv2.VideoWriter('D:/Research/2020ContrastiveLearningForSceneLabel/Paper/video/similarity.avi',fourcc,30.0,(int(1024.0*ratio),int(768.0*ratio)))


# video.set(cv2.CAP_PROP_POS_FRAMES, start)

# while(1):
#     fno = video.get(cv2.CAP_PROP_POS_FRAMES)
#     if fno > end:
#         break
#     ret, frame = video.read()

#     frame = cv2.resize(frame,(int(1024.0*ratio),int(768.0*ratio)))
#     out.write(frame)
#     # cv2.imshow('img', frame)
#     # cv2.waitKey(0)

# out.release()
# exit(0)



query_feat = feat[start]
query_feat = query_feat[:,np.newaxis]
dises = np.matmul(feat, query_feat)
dises = dises[start:end,0]

# np.savetxt('similarity.txt',dises,'%.3f')




fig = plt.figure()
ax = fig.add_subplot(111)
img = gImage.imread(pkuBirdViewImg)
img = img[0:1087,:,:]
img[:,:,3] = alpha
ax.imshow(img, zorder = 0)
ax.scatter(GNSS[start:end,1], GNSS[start:end,0], s=1, c='black', zorder = 1) # 车辆行驶轨迹
# ax.scatter(GNSS[:,1], GNSS[:,0], s=1, c='black', zorder = 1) # 车辆行驶轨迹
plt.title('trajectory on map')
plt.axis('equal')
plt.show()

exit(0)

######### 直方图 ############

ds_path = 'D:/Research/2020ContrastiveLearningForSceneLabel/Paper/video/ds'
sim_hist_feat_path = featFile[0:-4] + '_simHistFeat.npy'
sim_hist_feat = np.load(sim_hist_feat_path)




# ===================计算scene descriptor========================
straight_seed_hist = sim_hist_feat[5589]
turn_seed_hist = sim_hist_feat[24708]
dyna_seed_hist = sim_hist_feat[27768]

# straight_scene_descriptor = []
# turn_scene_descriptor = []
# dyna_scene_descriptor = []

# for i in range(36924):
#     cur_hist = sim_hist_feat[i]
#     js_s = JS_D(cur_hist, straight_seed_hist)
#     js_t = JS_D(cur_hist, turn_seed_hist)
#     js_d = JS_D(cur_hist, dyna_seed_hist)
#     if js_s < js_t and js_s < js_d:
#         straight_scene_descriptor.append(cur_hist)
#     elif js_t < js_s and js_t < js_d:
#         turn_scene_descriptor.append(cur_hist)
#     else:
#         dyna_scene_descriptor.append(cur_hist)

# straight_scene_descriptor = np.average(np.array(straight_scene_descriptor), axis=0)
# turn_scene_descriptor =  np.average(np.array(turn_scene_descriptor), axis=0)
# dyna_scene_descriptor =  np.average(np.array(dyna_scene_descriptor), axis=0)

train_gt_label_allFrame = gt_label_allFrame.copy()
# train_gt_label_allFrame[testing_frome_frame_id:] = 0

straight_sim_hist_feat = sim_hist_feat[train_gt_label_allFrame==1]
turn_sim_hist_feat = sim_hist_feat[train_gt_label_allFrame==2]
dyna_sim_hist_feat = sim_hist_feat[train_gt_label_allFrame==3]

straight_scene_descriptor = np.average(straight_sim_hist_feat, axis=0)
turn_scene_descriptor = np.average(turn_sim_hist_feat, axis=0)
dyna_scene_descriptor = np.average(dyna_sim_hist_feat, axis=0)

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# plt.bar(range(100), dyna_sim_hist_feat,width=1, align='edge')
# plt.ylim(0,0.04)
# ax.set_xticks([0,25,50,75,99])
# ax.set_xticklabels(['-1','-0.5','0','0.5','1'])

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# plt.savefig('D:/Research/2020ContrastiveLearningForSceneLabel/Paper/video/scp_at.png')
# plt.show()
# exit(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ratio = 1
width = 1024 + 400
height = 768 
out = cv2.VideoWriter('D:/Research/2020ContrastiveLearningForSceneLabel/Paper/video/classify.avi',fourcc,30.0,(width,height))

video.set(cv2.CAP_PROP_POS_FRAMES, start)

scp_sr = 'D:/Research/2020ContrastiveLearningForSceneLabel/Paper/video/scp_sr.png'
scp_tr = 'D:/Research/2020ContrastiveLearningForSceneLabel/Paper/video/scp_tr.png'
scp_at = 'D:/Research/2020ContrastiveLearningForSceneLabel/Paper/video/scp_at.png'
scp_sr = cv2.imread(scp_sr)
scp_tr = cv2.imread(scp_tr)
scp_at = cv2.imread(scp_at)
scp_sr = scp_sr[0:427,100:,:]
scp_sr = cv2.resize(scp_sr,(300,200))
scp_tr = scp_tr[0:427,100:,:]
scp_tr = cv2.resize(scp_tr,(300,200))
scp_at = scp_at[0:427,100:,:]
scp_at = cv2.resize(scp_at,(300,200))

while(1):
    fno = video.get(cv2.CAP_PROP_POS_FRAMES)
    fno = int(fno)

    cur_hist = sim_hist_feat[fno]
    js_s = JS_D(cur_hist, straight_scene_descriptor)
    js_t = JS_D(cur_hist, turn_scene_descriptor)
    js_d = JS_D(cur_hist, dyna_scene_descriptor)


    ds_file_name = str(fno).zfill(10) + '.png'
    ds_file_name = os.path.join(ds_path, ds_file_name)
    ds = cv2.imread(ds_file_name)
    ds = ds[0:427,100:,:]
    ds = cv2.resize(ds,(300,200))

    if fno > end:
        break
    context = np.zeros((height, width ,3), np.uint8)
    ret, frame = video.read()

    # 把当前帧图像放到视频里
    context[0:768, 0:1024, :] = frame 

    # 把当前帧DS放到视频里
    y0 = 0
    y1 = np.shape(ds)[0] + y0
    x0 = np.shape(frame)[1] - np.shape(ds)[1]
    x1 = np.shape(frame)[1]
    context[y0:y1,x0:x1,:] = ds

    ############################### 把各个scp放到视频里 ##############################
    interval = 84
    text_horizon_shift = 110
    text_vertical_shift = -130
    # scp straight road
    x0 = 1110
    x1 = x0 + np.shape(scp_sr)[1]

    y0 = 0
    y1 = np.shape(scp_sr)[0] + y0
    context[y0:y1,x0:x1,:] = scp_sr
    cv2.putText(context, '%.2f'%js_s,(x0+text_horizon_shift, y1+text_vertical_shift),1,2,(0,0,0),2)

    y0 = y1 + interval
    y1 = np.shape(scp_sr)[0] + y0
    context[y0:y1,x0:x1,:] = scp_tr
    cv2.putText(context, '%.2f'%js_t,(x0+text_horizon_shift, y1+text_vertical_shift),1,2,(0,0,0),2)

    y0 = y1 + interval
    y1 = np.shape(scp_sr)[0] + y0
    context[y0:y1,x0:x1,:] = scp_at
    cv2.putText(context, '%.2f'%js_d,(x0+text_horizon_shift, y1+text_vertical_shift),1,2,(0,0,0),2)

    p1y = 0
    if js_s < js_t and js_s < js_d: 
        p1y = p1y
    elif js_t < js_s and js_t < js_d:
        p1y = p1y + np.shape(scp_sr)[0] + interval
    else:
        p1y = p1y + (np.shape(scp_sr)[0] + interval) * 2
    p1x = 1110
    p2x = p1x + np.shape(scp_sr)[1]
    p2y = p1y + np.shape(scp_sr)[0]
    thickness = 4
    cv2.rectangle(context, (p1x,p1y+thickness),(p2x, p2y-thickness),(0,0,255),thickness=thickness*2)

    cv2.imshow('context',context)
    cv2.waitKey(1)
    out.write(context)

out.release() 

exit(0)
############################################

for i in range(start, end+1):
    ds_file_name = str(i).zfill(10)
    ds_file_name = os.path.join(ds_path, ds_file_name)
    cur_hist = sim_hist_feat[i]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.bar(range(100), cur_hist,width=1, align='edge')
    plt.ylim(0,0.06)
    ax.set_xticks([0,25,50,75,99])
    ax.set_xticklabels(['-1','-0.5','0','0.5','1'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(ds_file_name)
    plt.close()
    print(i)

# plt.show()
exit()

