#
# Filename: warp_template.py
# Author: Jun Xu
# Email: junx@cs.bu.edu
# Created Time: Thu 03 Dec 2015 05:16:45 PM EST
# Description: This script is to transform the base template. The transformation
#		includes in-/off-plane rotation and scaling. As a result, this script
#		outputs various transformed templates to "template_warped" under "results"
#		directory.
#

#-------------------------------------------------------------------------------
# import
#-------------------------------------------------------------------------------

import cv2
import numpy

from math import cos
from math import pi
from math import sin
from math import sqrt
from os import makedirs
from os.path import exists

dirname = '../../results/template_trans/'
if not exists(dirname):
	makedirs(dirname)

#-------------------------------------------------------------------------------
# put the template to a larger frame so that it will still be within the frame
# after transformation
#-------------------------------------------------------------------------------

fname = '../../resources/template.jpg'
tpl = cv2.imread(fname, cv2.CV_LOAD_IMAGE_COLOR)
tpl = cv2.resize(tpl, (0,0), fx=0.1, fy=0.1)

rows, cols, _ = tpl.shape
half_rows = int(rows / 2)
half_cols = int(cols / 2)

length = int(sqrt(pow(rows, 2) + pow(cols, 2)))
half_length = int(length / 2)

r_min = half_length - half_rows
c_min = half_length - half_cols
r_max = half_length + rows - half_rows
c_max = half_length + cols - half_cols

base = numpy.zeros((length, length, 3), numpy.uint8)
base[r_min:r_max, c_min:c_max] = tpl

#-------------------------------------------------------------------------------
# transform the template, ordered by in-plane rotation, off-plane rotation and
# scaling
#-------------------------------------------------------------------------------

scale_size = 6
angle_size = 7

log_scale = numpy.linspace(-1.0, 1.0, scale_size)
scale = [ pow(2, s) for s in log_scale ]
angle_in = numpy.linspace(-45, 45, angle_size)
angle_off = numpy.linspace(-45, 45, angle_size)

f = 1000
dx = 0
dy = 0
dz = f

A1 = numpy.array([
	[ 1, 0, -half_length ],
	[ 0, 1, -half_length ],
	[ 0, 0, 0 ],
	[ 0, 0, 1 ]
	])
T = numpy.array([
	[ 1, 0, 0, dx ],
	[ 0, 1, 0, dy ],
	[ 0, 0, 1, dz ],
	[ 0, 0, 0, 1 ]
	])
A2 = numpy.array([
	[ f, 0, half_length, 0 ],
	[ 0, f, half_length, 0 ],
	[ 0, 0, 1, 0 ]
	])

for i,ai in enumerate(angle_in):
	M_in_rotate = cv2.getRotationMatrix2D((half_length, half_length), ai, 1)
	tpl_in_rotate = cv2.warpAffine(base, M_in_rotate, (length, length))

	for j,ao in enumerate(angle_off):
		ao *= pi / 180.0
		R = numpy.array([
			[ cos(ao), 0, -sin(ao), 0 ],
			[ 0, 1, 0, 0 ],
			[ sin(ao), 0, cos(ao), 0 ],
			[ 0, 0, 0, 1 ]
			])
		M_off_rotate = numpy.dot(A2, numpy.dot(T, numpy.dot(R, A1)))
		tpl_off_rotate = cv2.warpPerspective(tpl_in_rotate, M_off_rotate, (length, length))

		for k,s in enumerate(scale):
			print('processing ai | ao | s: %0.f | %0.3f | %0.3f' % (ai, ao, s))

			tpl_scaling = cv2.resize(tpl_off_rotate, (0,0), fx=s, fy=s)
			fname = dirname + str(i) + '-' + str(j) + '-' + str(k) + '.jpg'
			cv2.imwrite(fname, tpl_scaling)
