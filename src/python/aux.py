#
# Filename: aux.py
# Author: Jun Xu
# Email: junx@cs.bu.edu
# Created Time: Sat 12 Dec 2015 09:56:57 PM EST
# Description: This script computes the matching score (namely, normalized
#		correlation coefficient and mutual information) between the averaged left
#		ear image and right ear image.
#

#-------------------------------------------------------------------------------
# import
#-------------------------------------------------------------------------------

import cv2
import numpy

from math import sqrt

#-------------------------------------------------------------------------------
# compute the Normalized Correlation Coefficient (NCC) between two images
#
# @l:			the averaged image of the left ear
# @r:			the averaged image of the right ear
# @mask:	the mask associated with the base template, used to find the ROI
#-------------------------------------------------------------------------------
def compute_ncc(l, r, mask):
	L_mean = cv2.mean(l, mask)
	R_mean = cv2.mean(r, mask)

	L = cv2.subtract(l, L_mean, mask=mask, dtype=cv2.CV_32S)
	R = cv2.subtract(r, R_mean, mask=mask, dtype=cv2.CV_32S)

	L2 = cv2.pow(L, 2)
	R2 = cv2.pow(R, 2)

	L2_sum = cv2.sumElems(L2)
	R2_sum = cv2.sumElems(R2)

	LR = cv2.multiply(L, R)
	LR_sum = cv2.sumElems(LR)

	return LR_sum[0] / sqrt(L2_sum[0] * R2_sum[0])

#-------------------------------------------------------------------------------
# compute the histogram range of the values in the array img
# cited from http://pythonhosted.org/MedPy/_modules/medpy/metric/image.html
#
# @img:		an image, represented as numpy.ndarray
# @bins:	number of bins in the histogram
#-------------------------------------------------------------------------------
def __range(a, bins):
	a = numpy.asarray(a)
	a_max = a.max()
	a_min = a.min()
	s = 0.5 * (a_max - a_min) / float(bins - 1)
	return (a_min - s, a_max + s)

#-------------------------------------------------------------------------------
# compute the entropy of the flattened data set (e.g. a density distribution)
# cited from http://pythonhosted.org/MedPy/_modules/medpy/metric/image.html
#
# @hist: a histogram of a grayscale image
#-------------------------------------------------------------------------------
def __entropy(hist):
	hist = hist[numpy.nonzero(hist)]
	hist = hist / float(numpy.sum(hist))
	log_hist = numpy.log(hist)
	prod = numpy.multiply(hist, log_hist)
	prod = numpy.nan_to_num(prod)
	return -1. * sum(prod)

#-------------------------------------------------------------------------------
# compute the mutual information between two images
# cited from http://pythonhosted.org/MedPy/_modules/medpy/metric/image.html
#
# @l:	the averaged left ear image
# @r:	the averaged right ear image
#-------------------------------------------------------------------------------
def compute_mi(l, r):
	bins=256

	l_range = __range(l, bins)
	r_range = __range(r, bins)

	l_hist, _ = numpy.histogram(l, bins=bins, range=l_range)
	r_hist, _ = numpy.histogram(r, bins=bins, range=r_range)
	lr_hist, _, _ = numpy.histogram2d(l.flatten(), r.flatten(), bins=bins, \
			range=[l_range, r_range])

	l_hist[0] = 0
	r_hist[0] = 0
	lr_hist[0] = 0

	l_entropy = __entropy(l_hist)
	r_entropy = __entropy(r_hist)
	lr_entropy = __entropy(lr_hist)

	return l_entropy + r_entropy - lr_entropy 
