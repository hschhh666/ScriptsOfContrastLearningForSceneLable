import os
import cv2

anchor_path = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/RingroadTestData/labeledData/CampusSceneDataset_labeldByTurn2/anchorImgs'
label_path = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/RingroadTestData/labeledData/CampusSceneDataset_labeldByTurn2/label.txt'

imgs = []
for _,_, imgs in os.walk(anchor_path):
    pass
imgs = [os.path.join(anchor_path, i) for i in imgs]

f = open(label_path, 'w')

anchor_num = len(imgs)
labels = [0 for i in range(anchor_num)]

i = 0
while i != anchor_num:
    img = imgs[i]
    img = cv2.imread(img)
    label = labels[i]
    img = cv2.putText(img, str(label) , (50, 50), 0, 1.2, (0, 0, 255), 2)


    cv2.imshow('img', img)
    label = chr(cv2.waitKey(0))
    if label == '1' or label == '2' or label == '3':
        labels[i] = label
        i += 1
    else:
        i -= 1
    
for i in labels:
    f.write(i+'\n')
f.close()