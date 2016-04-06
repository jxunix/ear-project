#
# Filename: detect_asm.py
# Author: Jun Xu
# Email: junx@cs.bu.edu
# Created Time: Thu 17 Mar 2016 03:07:46 PM EDT
# Description: This script is to detect ear regions by applying active shape
#   model. The input data are the images of those who had 3 visits.
#

#-------------------------------------------------------------------------------
# import
#-------------------------------------------------------------------------------

import cv2
import numpy
import image
import template

from os import listdir
from os import makedirs
from os import remove
from os.path import exists
from os.path import isfile
from os.path import join

#-------------------------------------------------------------------------------
# main body
#-------------------------------------------------------------------------------

print('Listing all images...')
img_dname = '../../outputs/4_flipped/'
img_fnames = [ f for f in listdir(img_dname) if isfile(join(img_dname, f)) and
		'-visit3-' in f ]

img_fnames_cpy = []
for f in img_fnames:
	before = f[0:3]
	after = f[11:]
	img_fnames_cpy.append(f)
	img_fnames_cpy.append(before + '-visit1-' + after)
	img_fnames_cpy.append(before + '-visit2-' + after)

img_fnames = [ f for f in img_fnames_cpy if isfile(join(img_dname, f)) ]
img_fnames.sort()

crv_fname = '../../asm/models/ear_99pts.crvs'
pt_dname = '../../asm/points_asm/'
dname = '../../outputs/detected_asm/'

if not exists(dname):
	makedirs(dname)

with open(crv_fname) as f:
	crvs = [ line.split() for i,line in enumerate(f) if i >= 1 and i <= 3 ]
crvs = [ c[8:-2] for c in crvs ]
crvs = [[ int(e) for e in c ] for c in crvs ]

for i,f in enumerate(img_fnames):
	print('Drawing the feature points for image %s...' % f)
	img = cv2.imread(img_dname + f, cv2.CV_LOAD_IMAGE_COLOR)

	pt_fname = f[:13] + '.pts'
	if not exists(join(pt_dname, pt_fname)):
		pt_fname = f[:13] + ' (fail).pts'
		if not exists(join(pt_dname, pt_fname)):
			pt_fname = f[0:13] + ' (remedy).pts'
	
	with open(pt_dname + pt_fname) as pt_f:
		pts = [ line.split() for j,line in enumerate(pt_f) if j >= 3 and j <= 101 ]
		pts = [ int(float(e)) for pt in pts for e in pt ]

		pts = numpy.array(pts, numpy.int32)
		pts = pts.reshape((-1,1,2))
		
		for c in crvs:
			pts_sub = pts[c]
			cv2.polylines(img, [pts_sub], False, (0,255,0), 20)

		img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
		print('  Outputing the ear regions...')
		cv2.imwrite(dname + f, img)
		# cv2.imshow('test', img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
