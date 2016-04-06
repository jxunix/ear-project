#
# Filename: image.py
# Author: Jun Xu
# Email: junx@cs.bu.edu
# Created Time: Fri 04 Dec 2015 12:35:49 PM EST
# Description: This script represents the Image class
#

#-------------------------------------------------------------------------------
# import
#-------------------------------------------------------------------------------

import cv2
import numpy
import template

from math import cos
from math import pi
from math import sin
from math import sqrt
from os.path import isfile

#-------------------------------------------------------------------------------
# constants
#-------------------------------------------------------------------------------

f = 1000
dx = 0
dy = 0
dz = f

T = numpy.array([
	[ 1, 0, 0, dx ],
	[ 0, 1, 0, dy ],
	[ 0, 0, 1, dz ],
	[ 0, 0, 0, 1 ]
	])

#-------------------------------------------------------------------------------
# class definition
#
# @bgr:			the BGR image
# @gray:		the grayscale image
# @rows:		the number of rows of the image
# @cols:		the number of cols of the image
#
# @result:	the matching result matrix
#-------------------------------------------------------------------------------

class Image:
	#-----------------------------------------------------------------------------
	# initialization function
	#
	# @fname: the filename of the image
	#-----------------------------------------------------------------------------

	def __init__(self, fname):
		self.bgr = cv2.imread(fname, cv2.CV_LOAD_IMAGE_COLOR)
		self.bgr = cv2.resize(self.bgr, (0,0), fx=0.1, fy=0.1)

		self.gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
		self.gray = cv2.equalizeHist(self.gray)
		self.rows, self.cols, _ = self.bgr.shape

	#-----------------------------------------------------------------------------
	# template matching #
	# @tpl:	a template object
	#-----------------------------------------------------------------------------

	def find_match(self, tpl):
		w = tpl.bbox_param['w']
		h = tpl.bbox_param['h']
		T = tpl.T
		T2_sum = tpl.T2_sum
		mask = tpl.mask

		rows = self.rows - h + 1
		cols = self.cols - w + 1

		if rows <= 0 or cols <= 0:
			return { 'ncc': 0, 'loc': (0,0) }
		else:
			result = numpy.zeros((rows, cols), dtype=float)

			for i in range(rows):
				for j in range(cols):
					I = self.gray[i:i+h, j:j+w]
					I_mean = cv2.mean(I, mask)
					I = cv2.subtract(I, I_mean, mask=mask, dtype=cv2.CV_32S)

					I2 = cv2.pow(I, 2)
					I2_sum = cv2.sumElems(I2)

					IT = cv2.multiply(I, T)
					IT_sum = cv2.sumElems(IT)

					result[i,j] = IT_sum[0] / sqrt(I2_sum[0] * T2_sum[0])

			_, max_val, _, max_loc = cv2.minMaxLoc(result)

			# print('ncc=%0.4f @%s' % (max_val, max_loc))
			return { 'ncc': max_val, 'loc': max_loc }

	#-----------------------------------------------------------------------------
	# find the ROI defined by the template
	#
	# @tpl:	a template object
	# @loc: the coordinate of the top-left corner of the template that matches the
	#		image best
	#-----------------------------------------------------------------------------

	def find_roi(self, tpl, loc):
		x,y = loc
		w = tpl.bbox_param['w']
		h = tpl.bbox_param['h']

		mask = numpy.zeros((self.rows, self.cols), numpy.uint8)
		mask[y:y+h, x:x+w] = tpl.mask

		blank = numpy.zeros(self.bgr.shape, numpy.uint8)
		roi = cv2.add(self.bgr, blank, mask=mask)

		return roi

	#-----------------------------------------------------------------------------
	# find the ROI defined by the template, display the matching area with the
	# boundary
	#
	# @tpl:	a template object
	# @loc: the coordinate of the top-left corner of the template that matches the
	#		image best
	#-----------------------------------------------------------------------------

	def find_matching_boundary(self, tpl, loc):
		x,y = loc
		w = tpl.bbox_param['w']
		h = tpl.bbox_param['h']

		mask = numpy.zeros((self.rows, self.cols), numpy.uint8)
		mask[y:y+h, x:x+w] = tpl.mask

		contours, hierarchy = cv2.findContours(mask, 1, 2)
		areas = [ cv2.contourArea(c) for c in contours ]
		maxArea = max(areas)
		index = [ i for i,a in enumerate(areas) if a == maxArea ]
		index = index[0]

		img = self.bgr.copy()
		cv2.drawContours(img, contours, index, (0,255,0), 2)

		return img

	#-----------------------------------------------------------------------------
	# find the standardized ROI defined by the template
	#
	# @tpl:			a template object
	# @loc:			the coordinate of the top-left corner of the template that matches the
	#		image best
	# @l_norm:	the length of the base template
	# @ai:			the in-plane rotation angle
	# @ao:			the off-plane rotation angle
	# @s:				the scaling factor
	#-----------------------------------------------------------------------------

	def find_roi_norm(self, tpl, loc, l_norm, ai, ao, s):
		l = tpl.length
		hl = int(l/2)

		x,y = loc
		w = tpl.bbox_param['w']
		h = tpl.bbox_param['h']

		roi = self.bgr[y:y+h, x:x+w]
		blank = numpy.zeros(roi.shape, numpy.uint8)
		roi = cv2.add(roi, blank, mask=tpl.mask)

		x = tpl.bbox_param['x']
		y = tpl.bbox_param['y']

		base = numpy.zeros((l,l,3), numpy.uint8)
		base[y:y+h, x:x+w] = roi

		ai = -ai
		s = 1.0 / float(s)
		ao *= pi / 180.0
		hl_norm = int(l_norm / 2)

		A1 = numpy.array([
			[ 1, 0, -hl_norm ],
			[ 0, 1, -hl_norm ],
			[ 0, 0, 0 ],
			[ 0, 0, 1 ]
			])
		A2 = numpy.array([
			[ f, 0, hl_norm, 0 ],
			[ 0, f, hl_norm, 0 ],
			[ 0, 0,	1, 0 ]
			])
		R = numpy.array([
			[ cos(ao), 0, -sin(ao), 0 ],
			[ 0, 1, 0, 0 ],
			[ sin(ao), 0, cos(ao), 0 ],
			[ 0, 0, 0, 1 ]
			])

		base = cv2.resize(base, (0,0), fx=s, fy=s)
		M = numpy.dot(A2, numpy.dot(T, numpy.dot(R, A1)))
		M = numpy.linalg.inv(M)
		base = cv2.warpPerspective(base, M, (l_norm, l_norm))
		M = cv2.getRotationMatrix2D((hl_norm, hl_norm), ai, 1)
		base = cv2.warpAffine(base, M, (l_norm, l_norm))

		return base
