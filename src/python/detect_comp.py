#
# Filename: detect_comp.py
# Author: Jun Xu
# Email: junx@cs.bu.edu
# Created Time: Mon 04 Apr 2016 01:20:46 PM EDT
# Description: This script is to compare the two detection methods, namely,
#   Active Shape Model (ASM) and Generalized Template Matching (GTM).
#   The metric for comparsion is intersection over union (IOU).
#

#-------------------------------------------------------------------------------
# import
#-------------------------------------------------------------------------------

import cv2
import numpy

from csv import writer
from os import listdir
from os import makedirs
from os import remove
from os.path import exists
from os.path import isfile
from os.path import join

#-------------------------------------------------------------------------------
# main body
#-------------------------------------------------------------------------------

img_dname = '../../outputs/4_flipped/'
truth_pt_dname = '../../asm/points_99/'
asm_pt_dname = '../../asm/points_asm/'
gtm_dname = '../../outputs/5_detected_template_matching/'

img_fnames = [ f for f in listdir(img_dname) if isfile(join(img_dname, f)) and 'remedy' not in f ]
img_fnames.sort()

# read the ear outer curve
crv_fname = '../../asm/models/ear_99pts.crvs'
with open(crv_fname) as f:
	crvs = [ line.split() for i,line in enumerate(f) if i == 1 ]
crv = crvs[0]
crv = crv[8:-2]
crv = [ int(e) for e in crv ]

iou_tbl = []

# for each image, find the ground-truth ROI and those detected by ASM and GTM,
# and compute the IOU for each detection method
for i,f in enumerate(img_fnames):
# if True:
	# i = 1; f = img_fnames[i]

	print('Processing image %s...' % f)

	# read the ear image
	img = cv2.imread(img_dname + f, cv2.CV_LOAD_IMAGE_COLOR)
	rows, cols, _ = img.shape

	# read the ground-truth feature points
	truth_pt_fname = f[:13] + '.pts'
	with open(truth_pt_dname + truth_pt_fname) as truth_pt_f:
		truth_pts = [ line.split() for j,line in enumerate(truth_pt_f) if j >= 3 and j <= 101 ]
		truth_pts = [ int(float(e)) for pt in truth_pts for e in pt ]

	truth_pts = numpy.array(truth_pts, numpy.int32)
	truth_pts = truth_pts.reshape((-1,1,2))

	# draw the polygon bounded by the ground-truth feature points
	roi_truth = numpy.zeros((rows, cols, 1), numpy.uint8)
	truth_polyline = truth_pts[crv]
	cv2.fillPoly(roi_truth, [truth_polyline], 255)
	roi_truth = cv2.resize(roi_truth, (0,0), fx=0.1, fy=0.1)

	# read the feature points detected by ASM
	asm_pt_fname = f[:13] + '.pts'
	if not exists(join(asm_pt_dname, asm_pt_fname)):
		asm_pt_fname = f[:13] + ' (fail).pts'

	with open(asm_pt_dname + asm_pt_fname) as asm_pt_f:
		asm_pts = [ line.split() for j,line in enumerate(asm_pt_f) if j >=3 and j <= 101 ]
		asm_pts = [ int(float(e)) for pt in asm_pts for e in pt ]

	asm_pts = numpy.array(asm_pts, numpy.int32)
	asm_pts = asm_pts.reshape((-1,1,2))

	# draw the polygon bounded by the feature points detected by ASM
	roi_asm = numpy.zeros((rows, cols, 1), numpy.uint8)
	asm_polyline = asm_pts[crv]
	cv2.fillPoly(roi_asm, [asm_polyline], 255)
	roi_asm = cv2.resize(roi_asm, (0,0), fx=0.1, fy=0.1)

	# find the ROI detected by GTM
	gtm = cv2.imread(gtm_dname + f, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	_, roi_gtm = cv2.threshold(gtm, 10, 255, cv2.THRESH_BINARY)

	# img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
	# cv2.imshow('image', img)
	# cv2.imshow('roi_truth', roi_truth)
	# cv2.imshow('roi_asm', roi_asm)
	# cv2.imshow('roi_gtm', roi_gtm)

	# compute IOU for ASM
	asm_intxn = cv2.bitwise_and(roi_truth, roi_asm)
	asm_union = cv2.bitwise_or(roi_truth, roi_asm)
	asm_ratio = float(cv2.countNonZero(asm_intxn)) / cv2.countNonZero(asm_union)

	# compute IOU for GTM
	gtm_intxn = cv2.bitwise_and(roi_truth, roi_gtm)
	gtm_union = cv2.bitwise_or(roi_truth, roi_gtm)
	gtm_ratio = float(cv2.countNonZero(gtm_intxn)) / cv2.countNonZero(gtm_union)

	print('IOU: asm = %0.1f%% vs gtm = %0.1f%%' % (asm_ratio * 100, gtm_ratio * 100))

	# cv2.imshow('asm_intxn', asm_intxn)
	# cv2.imshow('asm_union', asm_union)
	# cv2.imshow('gtm_intxn', gtm_intxn)
	# cv2.imshow('gtm_union', gtm_union)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	iou_tbl.append([ f, asm_ratio, gtm_ratio ])

# save the results to an external csv file
print('Outputing the IOU\'s of the two detection methods...')
csv_fname = '../../results/detection_comparision_iou.csv'
try:
	remove(csv_fname)
except OSError:
	pass

csv_file = open(csv_fname, 'wb')
csv_writer = writer(csv_file)
csv_writer.writerow([ 'Filename', 'ASM', 'GTM' ])
csv_writer.writerows(iou_tbl)
csv_file.close()
