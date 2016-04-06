#
# Filename: template.py
# Author: Jun Xu
# Email: junx@cs.bu.edu
# Created Time: Fri 04 Dec 2015 11:55:38 AM EST
# Description: This script represents the Template class.
#

#-------------------------------------------------------------------------------
# import
#-------------------------------------------------------------------------------

import cv2
import numpy

#-------------------------------------------------------------------------------
# class definition
#
# @bgr:			the BGR template
# @gray:		the grayscale template
# @mask:		a numpy 2d array associated with the template, 0 if the
#								corresponding	pixel in the template is black, 255 otherwise
# @length:	the length of the template (a square)
# @T:				the normalized template by subtracting the mean of unmasked	pixels
# @T2_sum:	the sum of the square of all normalized pixels
#
# @bbox:		the bounding box of the non-zero pixels
#-------------------------------------------------------------------------------

class Template:
	#-----------------------------------------------------------------------------
	# initialization function
	#
	# @fname:	the filename of the template
	#-----------------------------------------------------------------------------

	def __init__(self, fname):
		self.bgr = cv2.imread(fname, cv2.CV_LOAD_IMAGE_COLOR)
		self.gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
		_, self.mask = cv2.threshold(self.gray, 10, 255, cv2.THRESH_BINARY)
		self.gray = cv2.equalizeHist(self.gray)
		self.length = self.bgr.shape[0]

		self.find_bounding_box()

		T_mean = cv2.mean(self.bbox, self.mask)
		self.T = cv2.subtract(self.bbox, T_mean, mask=self.mask, dtype=cv2.CV_32S)
		T2 = pow(self.T, 2)
		self.T2_sum = cv2.sumElems(T2)
		
	#-----------------------------------------------------------------------------
	# find the bounding box of the non-zero pixels
	#-----------------------------------------------------------------------------

	def find_bounding_box(self):
		binary = numpy.copy(self.mask)
		contours, _ = cv2.findContours(binary,
				cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnt = contours[0]
		x,y,w,h = cv2.boundingRect(cnt)

		self.bbox = self.gray[y:y+h, x:x+w]
		self.mask = self.mask[y:y+h, x:x+w]
		self.bbox_param = { 'x': x, 'y': y, 'w': w, 'h': h }
