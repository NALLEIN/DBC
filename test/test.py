import cv2
from pathlib import Path

keys = []
meta_info_file = '/home/jianghao/Code/bytedance/DBC/codes/data/meta_info/clic.txt'
gt_root = Path('/media/jianghao/Samsung_T5/dataset/clic/all')
with open(meta_info_file, 'r') as fin:
    keys = [line.split('\n')[0] for line in fin]

img0 = cv2.imread('/media/jianghao/Samsung_T5/dataset/clic/all/adam-przewoski-193.png', cv2.IMREAD_COLOR)

for key in keys:
    # print(key)
    GT_path = str(gt_root / key)
    img = cv2.imread(GT_path, cv2.IMREAD_COLOR)
    # print(type(img))
    # print(GT_path)
    if not isinstance(img, type(img0)):
        print(key)