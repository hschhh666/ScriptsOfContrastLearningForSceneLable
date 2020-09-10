import pickle
import numpy as np
from cv2 import cv2
from utils import calculateDis
import os,shutil
import re


video_file = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\video.avi' # 视频文件
subtitle_file = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\video_subtitle.srt' # 字幕文件，里面保存了每帧图片对应的经纬度
posFile = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\pos.npy' # 把字幕文件里的经纬度提取出来保存成numpy数据，这样方便读取节约时间
anchor_img_path = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\labeledData\\test1' # 锚点照片路径
img_save_path = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\labeledData\\test4' # 锚点附近的照片路径


#======================创建保存照片的文件夹======================
if not os.path.exists(img_save_path): #目标文件夹如果不存在的话就创建
    os.mkdir(img_save_path)
else:
    key = input('Dictionary %s already exists!!! Press [y] to cover it, or press any key to exit : '%img_save_path)
    if key != 'y':
        exit(0)
    shutil.rmtree(img_save_path)
    os.mkdir(img_save_path)


#======================获取所有锚点并为它们分别创建文件夹======================
anchor_id = []
for root, dirs, files in os.walk(anchor_img_path): # 获取所有锚点
    files = files
    for f in files:
        match = re.match('(\d*)\.png',f)
        if match != None:
            anchor_id.append(int(match.group(1)))
anchor_id = sorted(anchor_id)
for i in anchor_id:
    os.mkdir(os.path.join(img_save_path,str(i))) # 为每个锚点创建文件夹

#======================参数配置======================

interval = 5 # 每间隔interval米选一张照片
sampleNum = 30 # 在每个锚点附近选择sampleNum张照片

#======================获取锚点附近的照片，并把它们的id和对应的路径保存在字典fno_to_path中======================
fno_to_path = {} # 创建字典，保存的是某帧图片对应的储存路径
GNSS = np.load(posFile)
for a in anchor_id:
    curPath = os.path.join(img_save_path, str(a))
    fno_to_path[a] = curPath
    # 从第i帧向前寻找sampleNum/2帧
    count = 0
    lastFrame = a
    curFrame = a
    while count < sampleNum/2:
        dis1, _ =  calculateDis(GNSS, lastFrame, curFrame)
        dis2, _ =  calculateDis(GNSS, lastFrame, curFrame + 1)
        if dis1 <= interval and dis2 >= interval:
            count += 1
            if abs(dis1 - interval) < abs(dis2 - interval):
                fno_to_path[curFrame] = curPath
                lastFrame = curFrame
            else:
                fno_to_path[curFrame + 1] = curPath
                lastFrame = curFrame + 1
        curFrame += 1
    # 从第i帧向后寻找sampleNum/2帧
    count = 0
    lastFrame = a
    curFrame = a
    while count < sampleNum/2 and curFrame - 1 >=0:
        dis1, _ =  calculateDis(GNSS, lastFrame, curFrame)
        dis2, _ =  calculateDis(GNSS, lastFrame, curFrame - 1)
        if dis1 <= interval and dis2 >= interval:
            count += 1
            if abs(dis1 - interval) < abs(dis2 - interval):
                fno_to_path[curFrame] = curPath
                lastFrame = curFrame
            else:
                fno_to_path[curFrame - 1] = curPath
                lastFrame = curFrame - 1
        curFrame -= 1    

#======================保存所有照片======================
video = cv2.VideoCapture(video_file)
fno = -1
img_num = len(fno_to_path)
count = 0
while 1:
    ret, img = video.read()
    fno += 1
    if not ret or count == img_num:
        print('Program exit normally.')
        exit(0)
    if fno in fno_to_path:
        img_name = os.path.join(fno_to_path[fno], str(fno)+'.png')
        ret = cv2.imwrite(img_name,img)
        if ret:
            count += 1
            print('Write image to %s [%d/%d]'%(img_name,count,img_num))
        else:
            print('Cannot write image to ' + img_name)
            print('Program exit.')
            exit(1)
        
exit(0)

# anchor_id = []
# sampleNum = 30
# id_to_class = {}
# skip = 5
# for i in tmp_anchor_id:
#     for j in range(i - int(sampleNum/2)*skip, i + int(sampleNum/2)*skip, skip):
#         if j < 0:
#             continue
#         anchor_id.append(j)
#         id_to_class[j] = i
# anchor_id = set(anchor_id)

# frame = 0
# while 1:
#     ret, img = origin_video.read()
#     if not ret :
#         exit(0)
#     if frame in anchor_id:
#         class_id = id_to_class[frame]
#         img_name = os.path.join(img_save_path, str(class_id))
#         if not os.path.exists(img_name):
#             os.mkdir(img_name)
#         img_name = os.path.join(img_name, str(frame)+'.png')
#         ret = cv2.imwrite(img_name, img)
#         if ret:
#             print('Write image to ' + img_name)
#         else:
#             print('Cannot write image to ' + img_name)
#             print('Program exit.')
#             exit(1)
#     frame = frame + 1




# frame = 0
# while 1:
#     ret, img = origin_video.read()
#     if not ret:
#         origin_video.release()
#         break
#     cv2.putText(img,'fno '+str(frame),(10,30),0,1,(0,0,255),2)
#     cv2.imshow('origin video', img)
#     key = cv2.waitKey(10)
#     if(key == 27):
#         cv2.destroyAllWindows()
#         break
#     if(key == 32):
#         img_name = os.path.join(img_save_path, str(frame)+'.png')
#         ret = cv2.imwrite(img_name, img)
#         if ret:
#             print('Write image to ' + img_name)
#         else:
#             print('Cannot write image to ' + img_name)
#             cv2.destroyAllWindows()
#             break
#     frame += 1

# print('hhh')