#
# Filename: flip.py
# Author: Jun Xu
# Email: junx@cs.bu.edu
# Created Time: Fri 26 Feb 2016 05:46:48 PM EST
# Description: This script flip all the right ears horizontally.
#

import cv2
import numpy

from os import listdir
from os import makedirs
from os.path import exists
from os.path import isfile
from os.path import join
from shutil import copyfile

dname_in = '../../outputs/3_remedied/'
dname_out = '../../outputs/4_flipped/'

fnames = [ f for f in listdir(dname_in) if isfile(join(dname_in, f)) ]

if not exists(dname_out):
	makedirs(dname_out)

for f in fnames:
	fname_in = dname_in + f
	fname_out = dname_out + f

	if '-r' in f:
		img = cv2.imread(fname_in, cv2.CV_LOAD_IMAGE_COLOR)
		img = cv2.flip(img, 1)
		cv2.imwrite(fname_out, img)
	else:
		copyfile(fname_in, fname_out)
