import os
import shutil
from glob import glob
from tqdm import tqdm

base_path = './TPE_results/tpe_tune'
fitness = os.listdir(base_path)

save_path = './tpe_results'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

for i in tqdm(range(len(fitness))):
    fit = fitness[i]
    dst_path = os.path.join(save_path, fit)
    src_path = os.path.join(base_path, fit, 'test')

    try:
        src_path = os.path.join(src_path, os.listdir(src_path)[0])
    except:
        continue

    files = glob(src_path + '/*.txt')

    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)

    for file in files:
        shutil.copy(file, dst_path)

print('haha')
#ln -sfb /home/users/feng01.gao/tianqi.shen/DenseSiam/tracking/tpe_results/* /home/users/feng01.gao/tianqi.shen/DenseSiam/tracking/result/UAV10FPS/
#ln -sfb /home/users/feng01.gao/tianqi.shen/DenseSiam/tracking/tpe_results/* /home/users/feng01.gao/tianqi.shen/SiameseTracking4UAV/results/UAV123_10fps/
