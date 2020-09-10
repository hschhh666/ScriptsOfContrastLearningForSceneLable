import cv2
import os

videoFile = '/home/hsc/Research/FewAnchorPointsBasedSceneLabeling/data/video.avi'
tarPath = '/home/hsc/Research/FewAnchorPointsBasedSceneLabeling/data/campusFrames/0'

if not os.path.isdir(tarPath):
        os.makedirs(tarPath)

video =  cv2.VideoCapture(videoFile)
fno = 0
while 1:
    ret,img = video.read()
    if not ret:
        break
    imgName = str(fno).rjust(10,'0') + '.png'
    path = os.path.join(tarPath,imgName)
    cv2.imwrite(path, img)
    fno += 1
    if fno == 1000:
        exit(0)
    if fno%10 == 0:
        print('Processing %d'%fno)

print('Finished. Program exit normally.')

