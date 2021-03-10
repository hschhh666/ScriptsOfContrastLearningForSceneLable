import numpy as np
import os
import cv2
import datetime
import time
import re
import shutil

# videoPath = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/RingroadTestData/video.avi'
videoPath = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/campus_img_dataset/video.avi'
savePath = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/RingroadTestData/labeledData/CampusSceneDataset_fuck'

pickAnchorImg = True

anchorImgPath = os.path.join(savePath, 'anchorImgs') # 保存的是锚点图像
datasetPath = os.path.join(savePath, 'dataset') # 保存的是包括及其相邻帧的图像，用于训练
if pickAnchorImg:
    ringRoadStartFrame = 36000#2897
    ringRoadEndFrame = 65857
    watiKeyDeley = 0 # 第一帧是暂停的
    bakupWaitKeyDeley = 41 # 1 ~ 100， 速度从慢到快
    playOrPause = True # 0 play 1 pause，第一帧是暂停的f
    maxSpeed = 100 # 其实这个控制的是最慢能有多慢，慢是没有下界的，但是快是有上界的，这个上界取决于对一帧图像的处理时间

    
    if not os.path.exists(anchorImgPath):
        os.makedirs(anchorImgPath)

    cap = cv2.VideoCapture(videoPath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ringRoadStartFrame)
    while(1):
        fno = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.rectangle(frame, (250,400),(1050,820),(0,0,255),2)
        # frame = frame[400:820,255:1055]
        originFrame = frame.copy()

        cv2.putText(frame, 'fno %d'%fno,(30,30), 2, 0.8, (0,0,255),2)
        cv2.putText(frame,'speed %d (1~100)'%(maxSpeed + 1 - bakupWaitKeyDeley), (30,60),2,0.8,(0,0,255),2)
        cv2.imshow('img',frame)
        key = cv2.waitKeyEx(watiKeyDeley)

        if key == 32: # 空格键，播放或暂停
            playOrPause = not playOrPause
            if not playOrPause:
                watiKeyDeley = bakupWaitKeyDeley
            else:
                bakupWaitKeyDeley = watiKeyDeley
                watiKeyDeley = 0
        elif key == 13: # Enter键，保存当前帧
            imgPath = str(int(fno)).rjust(10,'0') + '.png'
            imgPath = os.path.join(anchorImgPath, imgPath)
            res = cv2.imwrite(imgPath, originFrame)
            if res:
                print('Write anchor image to %s'%imgPath)
            else:
                raise ValueError('Error!!! Can not write anchor image to %s'%imgPath)
                exit(-1)
        elif key == 2490368: # 方向键 上， 加速播放
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            if not playOrPause:
                if watiKeyDeley <= 1:
                    watiKeyDeley = 1
                else:
                    watiKeyDeley = watiKeyDeley - 1
                bakupWaitKeyDeley = watiKeyDeley
            else:
                if bakupWaitKeyDeley <= 1:
                    bakupWaitKeyDeley = 1
                else:
                    bakupWaitKeyDeley = bakupWaitKeyDeley - 1
        elif key == 2621440: # 方向键 下， 减速播放
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            if not playOrPause:
                if watiKeyDeley < maxSpeed:
                    watiKeyDeley = watiKeyDeley + 1
                else:
                    watiKeyDeley = maxSpeed
                bakupWaitKeyDeley = watiKeyDeley
            else:
                if bakupWaitKeyDeley < maxSpeed:
                    bakupWaitKeyDeley = bakupWaitKeyDeley + 1
                else:
                    bakupWaitKeyDeley = maxSpeed
        elif key == 2424832: # 方向键 左， 后退
            fno = fno - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        elif key == 2555904: # 方向键 右， 前进
            fno = fno + 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        elif key == ord('f'):
            fno = fno + 10
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        elif key == ord('d'):
            fno = fno - 10
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        elif key == 27: # 退出
            print('Current fno =',fno)
            cap.release()


else:
    if os.path.exists(datasetPath):
        ans = input('Dataset path %s already exit, sure to remove and create a new one? [y]/n : '%datasetPath)
        if ans == 'n':
            print('Program exit.')
            exit(0)
        shutil.rmtree(datasetPath)
    imgNames = []
    for _,_, imgNames in os.walk(anchorImgPath):
        pass
    imgNames = [i[0:10] for i in imgNames]
    imgNamesInt = list(map(int, imgNames))
    sampleNum = 16
    sampleImgInt = []
    for i,_ in enumerate(imgNamesInt):
        if i == len(imgNamesInt) - 1:
            break
        start = imgNamesInt[i]
        end = imgNamesInt[i+1]
        tmp = list(np.linspace(start, end, sampleNum + 1).astype('int32'))[0:-1]
        sampleImgInt.append(tmp)
    
    cap = cv2.VideoCapture(videoPath)
    for oneAnchorImgs in sampleImgInt:
        folderPath = str(oneAnchorImgs[0]).rjust(10, '0')
        folderPath = os.path.join(datasetPath, folderPath)
        os.makedirs(folderPath)
        for imgFno in oneAnchorImgs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, imgFno)
            fno = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if fno != imgFno:
                raise ValueError('Frame number error!!! It should be %d, but it is %d'%(imgFno, fno))
            imgPath = str(imgFno).rjust(10,'0') + '.png'
            imgPath = os.path.join(folderPath, imgPath)
            ret, img = cap.read()
            if not ret:
                raise ValueError('Cannot read frame %d!'%imgFno)
            cv2.imwrite(imgPath, img)



    