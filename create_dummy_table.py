#!/usr/bin/env python3

import cv2
import numpy as np
import sys

imgpath = sys.argv[1]
table_path = sys.argv[2]
img = cv2.imread(imgpath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
scale = 0.005
orig = (-0.8, 1.6)
res = np.zeros((gray.size, 3), np.float32)
idx = 0
print(res.shape)
for y in range(gray.shape[0]):
    for x in range(gray.shape[1]):
        print("x: " + str(x) + " y: " + str(y))
        res[idx, 0] = float(x) * scale + orig[1]
        res[idx, 1] = float(y) * scale + orig[0]
        res[idx, 2] = gray[y, x]
        idx += 1
np.savetxt(table_path, res, delimiter=',', fmt='%f')