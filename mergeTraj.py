import os

TrajPath = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/RingroadTestData/traj'

TrajFiles = []

for _,_,TrajFiles in os.walk(TrajPath):
    TrajFiles = [os.path.join(TrajPath, f) for f in TrajFiles]

mergeFile = 'C:/Users/A/Desktop/merge.traj'
mergeF = open(mergeFile,'w')
num = 0
for fn in TrajFiles:
    mergeF.write('ThisIsFile%d\n'%num)
    num += 1
    f = open(fn)
    lines = f.readlines()
    mergeF.writelines(lines)

mergeF.close()
