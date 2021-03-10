import re
import matplotlib.pyplot as plt

filename = 'D:/Research/2020ContrastiveLearningForSceneLabel/Data/deepLearningRes/labeledByTurn2/log_20210120_22_19_09_lossMethod_nce_NegNum_8_Model_alexnet_lr_0.03_decay_0.0001_bsz_8_featDim_128_contrasMethod_e2e_traditionalMethod.txt'

losses = []
cnt = 0
with open(filename) as f:
    lines = f.readlines()
    for line in lines:
        match = re.match('.*loss \d+\.\d+ \((\d+\.\d+)\)', line)
        if match != None:
            cnt += 1
            if cnt == 1:
                cnt = 0
                losses.append(float(match.group(1)))

plt.plot(losses)
# plt.savefig(filename[0:-4]+'.pdf')
plt.show()