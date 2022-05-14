import time
import matplotlib.pyplot as plt 
import numpy as np 
files = ['log/imagenet2market-lw0.6/train.log',
         'log/imagenet2market_v2-lw0.6/train.log',
         'log/soft_triplet_imagenet2market-lw0.6/train.log',
         'log/imagenet2market-lw0.0/train.log']
colors = ['r.', 'y.', 'b.','g.']#, 'k.', 'g.']
labels = ['GCMT',
          'GCMTv2',
          'soft triplet loss',
          'baseline']
count = 0
while True:
    maps = []
    for fname in files:
        fid = open(fname, 'r')
        lines = fid.readlines()
        fid.close()
        temp_maps = []
        for line in lines:
            if 'best:' in line:
                # print(line)
                this_map = line.split('%')[-2].split()[-1]
                temp_maps.append(float(this_map))
        maps.append(temp_maps)

    min_len = 10000
    max_len = 0
    for i in range(len(maps)):
            max_len = max_len if max_len > len(maps[i]) else len(maps[i])
            min_len = min_len if min_len < len(maps[i]) else len(maps[i])
        #     plt.plot(np.array(maps[i]), colors[i])
        # else:
        #     plt.plot(np.array(maps[i]), colors[i])
            # plt.plot(np.array(maps[i][:max_len]), colors[i])
            # plt.plot(np.array(maps[i][:min_len]), colors[i])
    x = np.arange(1, min_len+1)
    for i in range(len(maps)):
        plt.plot(x, np.array(maps[i][:min_len]), colors[i], label=labels[i])

    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xlim((0.5, min_len+2))
    plt.savefig('accuracy.jpg')
    plt.close()
    if max_len == len(maps[-1]):
        break
    print(count)
    count += 1
    time.sleep(300)
    # break
