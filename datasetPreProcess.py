import os
import re
import numpy as np

# 检查数据集，确保每个锚点文件夹下至少有minSampleNum张图片。如果不够的话就复制文件夹下的图片凑够minSampleNum张。一个锚点文件夹下的照片编号是相邻的（可能相差5帧或者几帧），在凑照片时新加照片的标号首先要使用内部的标号，意思是如果某个文件夹下图片编号最小是12最大是99的话，新加的照片编号要先在12~99中间找，假如编号被用完了再考虑用该范围外的编号。这样做的目的是防止不同锚点文件夹的照片编号重叠，那问题可就大了。


datasetPath = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\campus_img_dataset\\labeledData\\test2\\train'
minSampleNum = 30

dirs = []
for root, d, files in os.walk(datasetPath):
    dirs = d
    print(dirs)
    break

for d in dirs:
    path = os.path.join(datasetPath, d)
    files = os.listdir(path)
    fileList = []
    for f in files:
        match = re.match('(\d+).png',f)
        f = int(match.group(1))
        fileList.append(f)
    fileList = sorted(fileList)
    num = len(fileList)

    print(num, d)
    continue

    if num >= minSampleNum:#如果当前锚点文件夹内文件的数量够了就不再处理了。
        continue
    left = minSampleNum - num
    curId = fileList[0]
    fileList = set(fileList)
    while left != 0:
        curId += 1
        if curId in fileList:# 如果这个编号没被使用，则用这个编号
            continue
        src = os.path.join(path, str(curId-1) + '.png')
        tar = os.path.join(path, str(curId) + '.png')
        os.system('copy ' + src + ' ' + tar)
        left = left - 1
