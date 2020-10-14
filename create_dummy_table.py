#!/usr/bin/env python3

import cv2
import numpy as np
import sys

imgpath = sys.argv[1]
table_path = sys.argv[2]
img = cv2.imread(imgpath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
np.savetxt(table_path, gray, delimiter=',', fmt='%d')