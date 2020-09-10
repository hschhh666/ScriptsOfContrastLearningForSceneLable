import re
import matplotlib.pyplot as plt

filename = 'D:\\Research\\2020ContrastiveLearningForSceneLabel\\Code\\Python\\datas\\20200831res\\20200831neg18featDim8e2e\\neg18feat8.txt'

losses = []
cnt = 0
with open(filename) as f:
    lines = f.readlines()
    for line in lines:
        match = re.match('.*loss \d+\.\d+ \((\d+\.\d+)\)', line)
        if match != None:
            cnt += 1
            if cnt == 15:
                cnt = 0
                losses.append(float(match.group(1)))

plt.plot(losses)
plt.savefig(filename[0:-4]+'.pdf')
plt.show()