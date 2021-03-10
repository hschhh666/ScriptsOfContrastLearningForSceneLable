import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

TrajPath = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/RingroadTestData/vehicleNum.txt'
videoPath = 'C:\\Users\\A\\Desktop/video.avi'
trafficSceneDataPath = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/trafficSceneData'
processedTrafficSceneDataPath = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/processedTrafficSceneData'

dirs = []
for root, dirs, _ in os.walk(trafficSceneDataPath):
    break

for d in dirs:
    os.makedirs(os.path.join(processedTrafficSceneDataPath, d))
    folderPath = os.path.join(trafficSceneDataPath, d)
    imgPaths = []
    for _, _, imgPaths in os.walk(folderPath):
        break
    if len(imgPaths) != 32:
        print('error!!!')
    for imgPath in imgPaths:
        scrPath = os.path.join(folderPath, imgPath)
        tarPath = os.path.join(processedTrafficSceneDataPath, d, imgPath)
        src = cv2.imread(scrPath)
        # cv2.rectangle(src, (100,400),(1200,820),(0,0,255),2)
        cropped = src[400:820,100:1200]
        cv2.imwrite(tarPath, cropped)
        # cv2.imshow('img', cropped)
        # cv2.waitKey(0)


        




exit(0)

f = open(TrajPath,'r')
lines = f.readlines()
timeStamp = []
vehicleNum = []

less = []
more = []
fno = -1
for line in lines:
    fno += 1
    t, n = map(int, line.split())
    timeStamp.append(t)
    vehicleNum.append(n)
    if n != -1 and n <= 5:
        less.append(fno)
    if n > 15:
        more.append(fno)

# plt.hist(vehicleNum, bins= 20)
# plt.plot()
# plt.show()

print(int(len(less)/640))
print(len(more)/640)
less = less[555:int(len(less)):int(len(less)/640)]
more = more[762:int(len(more)):int(len(more)/640)]

lessFolderCount = 40
moreFolderCount = 41
lessImgCount = 32
moreImgCount = 32
lessFolderPath = ''
moreFolderPath = ''

lessPath = os.path.join(trafficSceneDataPath,'less')
morePath = os.path.join(trafficSceneDataPath,'more')
cap = cv2.VideoCapture(videoPath)
fno = -1
while(1):
    fno += 1
    ret, frame = cap.read()
    if not ret:
        break
    if vehicleNum[fno] == -1:
        continue
    if fno in less:
        if lessImgCount == 32:
            lessFolderPath = os.path.join(trafficSceneDataPath, str(lessFolderCount).rjust(10,'0'))
            os.makedirs(lessFolderPath)
            lessImgCount = 0
            lessFolderCount += 2
        imgPath = os.path.join(lessFolderPath, str(fno).rjust(10, '0') + '.png')
        cv2.imwrite(imgPath, frame)
        lessImgCount += 1
    if fno in more:
        if moreImgCount == 32:
            moreFolderPath = os.path.join(trafficSceneDataPath, str(moreFolderCount).rjust(10,'0'))
            os.makedirs(moreFolderPath)
            moreImgCount = 0
            moreFolderCount += 2
        imgPath = os.path.join(moreFolderPath, str(fno).rjust(10, '0') + '.png')
        cv2.imwrite(imgPath, frame)
        moreImgCount += 1



    



















exit(0)

anchors = []
first = True
for i in range(len(vehicleNum)-1):
    if vehicleNum[i] == -1:
        if len(anchors) == 0:
            continue
        else:
            anchors.append(i)
            break
    if first:
        first = False
        anchors.append(i)
    if vehicleNum[i] < 11 and vehicleNum[i+1] >= 11:
        anchors.append(i)
    if vehicleNum[i] >= 11 and vehicleNum[i+1] < 11:
        anchors.append(i)


resAnchors = []
for i,_ in enumerate(anchors):
    if i == 0:
        resAnchors.append(anchors[i])
        continue
    anchorsLastIdx = len(resAnchors) - 1
    if anchorsLastIdx != 0 and anchors[i] - resAnchors[anchorsLastIdx] < 100:
        resAnchors.pop()
    else:
        resAnchors.append(anchors[i])


tmp = [11 for i,_ in enumerate(resAnchors)]
plt.scatter(resAnchors, tmp, c='r')

plt.plot(vehicleNum)
plt.show()

sampleImgIdx = []

for i in range(len(resAnchors) - 1):
    tmpSampleIdx = list(map(int, list(np.linspace(resAnchors[i], resAnchors[i+1]-1, 32))))
    sampleImgIdx.append(tmpSampleIdx)

cap = cv2.VideoCapture(videoPath)

fno = -1
anchorID = 0
sampleID = 0
anchorsPath = ''
while(True):
    fno += 1
    ret, frame = cap.read()
    if not ret:
        break
    if fno < sampleImgIdx[0][0]:
        continue
    if fno == sampleImgIdx[anchorID][0]:
        sampleID = 0
        anchorsPath =  str(fno).rjust(10,'0')
        anchorsPath = os.path.join(trafficSceneDataPath, anchorsPath)
        os.makedirs(anchorsPath)
    if fno == sampleImgIdx[anchorID][sampleID]:
        imgPath = str(fno).rjust(10,'0') + '.png'
        imgPath = os.path.join(anchorsPath, imgPath)
        cv2.imwrite(imgPath, frame)
        sampleID += 1
        if sampleID == len(sampleImgIdx[anchorID]):
            anchorID += 1
            sampleID = 0
            if anchorID == len(sampleImgIdx):
                break


cap.release()


exit(0)












avg = np.average(vehicleNum)
print(avg)

avgVehicleNum = []
frameWindowSize = 600
for i, _ in enumerate(vehicleNum):
    if i < frameWindowSize:
        continue
    tmpAvg = np.average(vehicleNum[i-frameWindowSize:i])
    avgVehicleNum.append(tmpAvg)

# plt.hist(vehicleNum, bins=20)
plt.plot([0,63000],[11,11])
plt.plot(avgVehicleNum)

plt.show()

pass
