#
# Filename: scale_raw.py
# Author: Jun Xu
# Email: junx@cs.bu.edu
# Created Time: Thu 17 Mar 2016 08:40:20 PM EDT
# Description: This script is to scale the raw images by 0.1.
#

import cv2
import numpy
from os import listdir
from os import makedirs
from os.path import exists
from os.path import isfile
from os.path import join

dname = '../../outputs/4_flipped/'
dname_out = '../../outputs/scaled_raw/'
fnames = [ f for f in listdir(dname) if isfile(join(dname, f)) and '-visit3-'
		in f ]

if not exists(dname_out):
	makedirs(dname_out)

fnames_cpy = []
for f in fnames:
	before = f[0:3]
	after = f[11:]
	fnames_cpy.append(f)
	fnames_cpy.append(before + '-visit1-' + after)
	fnames_cpy.append(before + '-visit2-' + after)

fnames = [ f for f in fnames_cpy if isfile(join(dname, f)) ]
fnames.sort()

for f in fnames:
	print('Scaling image %s...' % f)
	img = cv2.imread(dname + f, cv2.CV_LOAD_IMAGE_COLOR)
	img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
	cv2.imwrite(dname_out + f, img)
