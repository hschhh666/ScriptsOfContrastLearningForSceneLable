import numpy as np
from cv2 import cv2
import re
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
import torchvision.datasets as datasets
from utils import *
import random
from sklearn.cluster import KMeans
from sklearn import decomposition


datasetPath = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/RingroadTestData/labeledData/CampusSceneDataset_labeldByTurnAndDynamicTraffic/anchorImgs' # 数据集路径
srtSrc = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/campus_img_dataset/test.srt'
srtTar = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/campus_img_dataset/video_subtitle_withFno_withAnchorAndCalLabel_labeldByTurnAndDynamicTraffic.srt'
cal_label_allFrame_path = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/campus_img_dataset/cal_label_allFrame.txt'


anchorIdx = []
for _,_,anchorIdx in os.walk(datasetPath):
    pass

anchorIdx = [int(i[0:-4]) for i in anchorIdx]
f = open(srtSrc, 'r')
srcLines = f.readlines()
f.close()
f = open(cal_label_allFrame_path,'r')
cal_label_allFrame = f.readlines()
f.close()
f = open(srtTar, 'w')
fno = 0
cnt = 0
last = 0
res = []
for line in srcLines:
    if line[0] == 'f':
        f.write(line)
        f.write('label = ')
        f.write(cal_label_allFrame[fno])
        fno += 1
        if int(line[4:]) in anchorIdx:
            cnt += 1
            f.write('**********Anchor**********')
            res.append(last - int(line[4:]))
            last = int(line[4:])
        elif cnt != 0:
            cnt += 1
            if cnt == 10:
                cnt = 0
            f.write('**********Anchor**********')
    else:
        f.write(line)
f.close()

res = np.array(res)
exit()
mode = input()
key = int(input())
msg = input()
if mode.lower()=='d' or mode.lower()=='decrypt':
    key=-key
elif mode.lower()!='e' or mode.lower()!='encrypt':
    print('Wrong Mode')
    exit()
trmsg=''
for i in msg:
    if msg.isalpha():
        n=ord(i)+key
        if msg.isupper():
            if n>ord('Z'):
                n-=26
            if n<ord('A'):
                n+=26
        elif msg.islower():
            if n > ord('z'):
                n -= 26
            if n < ord('a'):
                n += 26
        trmsg+=chr(n)
    elif msg.isnumeric():
        n = ord(i)
        n += key%10
        if n > ord('9'):
            n-=10
        elif n<ord('0'):
            n+=10
        trmsg+=chr(n)
    else:
        trmsg+=i
print(trmsg)


exit(0)

grounp=int(input())
totalresult=[]
for i in range(grounp):
    n=int(input())
    goodsgrounp = []
    zidian = {}
    for j in range(n):
        goods=input().split()
        goodsgrounp.append(goods)
        for k in goodsgrounp:
            l=len(k)-1
            averagescore=sum(int(k[h]) for h in range(1,l+1))/int(l)
            zidian[k[0]]=(averagescore,l)
    result=sorted(zidian.items(),key=lambda item:(item[1][0],item[1][1]),reverse=True)
    for w in result:
        totalresult.append(w)
for v in totalresult:
    print(v[0])

exit(0)
a=[1,2,3]
print(id(a))
a=a[0]
print(id(a))
exit(0)
b = [4,5,6]
print(id(b))
for i in [a,b]:
    print(id(i))
    i=1
    print(id(i))
print(a)
print(b)



exit()
t = np.linspace(0, 2 * np.pi, 20)
x = np.sin(t)
y = np.cos(t)

plt.scatter(t,x,c=y)
plt.show()


exit(0)

print('hhh')
a = torch.randperm(8).long()
b = torch.zeros(8).long()
value = torch.arange(8)
b.index_copy_(0,a,value)

exit(0)
memPath = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/deepLearningRes/mem.npy'

feat = np.load(memPath)
#======================计算锚点与正负样本间的平均距离======================
loop = 30
avg_pos_dis = 0
avg_neg_dis = 0

idx = list(range(1280))

for i in range(loop):
    imgs_idx = [random.sample(idx[32*i:32*(i+1)],1)[0] for i in range(40)]
    imgs_feat = feat[imgs_idx] # 获取所有样本点的特征

    pos_idx = [random.sample(idx[32*i:32*(i+1)],1)[0] for i in range(40)]
    pos_feat = feat[pos_idx]
    pos_feat = pos_feat.transpose(1,0)
    matrix = np.matmul(imgs_feat, pos_feat)
    matrix = np.exp(matrix)
    pos_dis = matrix.trace()/len(imgs_idx)
    avg_pos_dis += pos_dis

    neg_idx = [random.sample(idx[32*(i+1):32*(i+2)],1)[0] for i in range(39)]
    neg_idx.append(5)
    neg_feat = feat[neg_idx]
    neg_feat = neg_feat.transpose(1,0)
    matrix = np.matmul(imgs_feat, neg_feat)
    matrix = np.exp(matrix)
    neg_dis = matrix.trace()/len(imgs_idx)
    avg_neg_dis += neg_dis
print('anchor to positive average distance is %f, to negative is %f'%(avg_pos_dis/loop, avg_neg_dis/loop))


kmeans = KMeans(n_clusters=2).fit(feat)
if np.shape(feat)[1]!= 2:
    pca = decomposition.PCA(n_components=2)
    pca.fit(feat)
    feat = pca.fit_transform(feat)


count = 0
if kmeans.labels_[0] == 0:
    c = ['r', 'g']
else:
    c = ['g', 'r']

for j in range(32):
    start = j*32
    end = (j+1)*32
    plt.figure()
    for i, v in enumerate(kmeans.labels_):
        if start <=i < end:
            continue
        plt.scatter(feat[i,0], feat[i,1], c = c[kmeans.labels_[i]])
        print(kmeans.labels_[i],end='')
        count += 1
        if count == 32:
            print()
            count = 0

    for i, v in enumerate(kmeans.labels_):
        if start <=i < end:
            plt.scatter(feat[i,0], feat[i,1], c = 'b')
    plt.savefig('D:/Research/2020ContrastiveLearningForSceneLabel/Data/cluster/2/%d.png'%int(start/32))
    plt.close()

plt.show()
exit(0)

class ImageFolderInstance(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

data_folder = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/processedTrafficSceneData'


train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

train_dataset = ImageFolderInstance(data_folder, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle= False)

for idx,(img, target, index) in enumerate(train_loader):
    pass


exit(0)

imgPath = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/processedTrafficSceneData/0000000001/0000010571.png'
img = Image.open(imgPath)

data1 = transforms.RandomResizedCrop(224, scale=(1., 1.))(img)
data2 = transforms.RandomResizedCrop(224, scale=(1., 1.))(img)
data3 = transforms.RandomResizedCrop(224, scale=(1., 1.))(img)
plt.subplot(2,2,1),plt.imshow(img),plt.title("0")
plt.subplot(2,2,2),plt.imshow(data1),plt.title("1")
plt.subplot(2,2,3),plt.imshow(data2),plt.title("2")
plt.subplot(2,2,4),plt.imshow(data3),plt.title("3")
plt.show()
exit(0)


x = np.linspace(-1,1)
y = np.exp(x)
plt.plot(x,y)
plt.show()


exit(0)


subtitleFile = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\video_subtitle.srt'
new_subtitleFile = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Data\\campus_img_dataset\\video_subtitle_withFno.srt'

with open(subtitleFile) as f:
    lines = f.readlines()

f = open(new_subtitleFile,'w')
fno = 0
for line in lines:
    if line != '\n':
        f.write(line)
    else:
        f.write('fno:'+str(fno)+'\n')
        f.write(line)
        fno += 1



exit(0)

size = 100
data = np.ones([size,size])

for i,_ in enumerate(data):
    for j,_ in enumerate(data[i]):
        dis = abs(i-j)
        coeff = 0.0001
        data[i][j] = np.exp(-dis*coeff)


data = [[0.9,0.9,0.9,0.9],[0.5,0.5,0.5,0.5],[0,0,0,0]]

plt.imshow(data, interpolation='nearest', cmap=plt.cm.Greys)
plt.show()

print()