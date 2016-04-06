#
# Filename: score_batch.py
# Author: Jun Xu
# Email: junx@cs.bu.edu
# Created Time: Sat 12 Dec 2015 09:42:11 PM EST
# Description: This script computes the matching scores between the averaged
#		left ear image and right ear image and outputs the values to
#		'results/matching_ncc_v#.csv' and 'results/matching_mi_v#.csv'
#		respectively.
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
from aux import compute_ncc
from aux import compute_mi

#-------------------------------------------------------------------------------
# constants
#-------------------------------------------------------------------------------

l_norm = 251

#-------------------------------------------------------------------------------
# main body
#-------------------------------------------------------------------------------
dname = '../../outputs/6_detected_template_matching_standardized/'
filenames = [ f for f in listdir(dname) if isfile(join(dname, f)) and '-l1' in f ]

visit_nums = list(xrange(3))
for visit_num in visit_nums:
	visit_num += 1

	fnames = [ f for f in filenames if 'visit' + str(visit_num) in f ]
	fnames.sort()
	
	img = cv2.imread(dname + fnames[0], cv2.CV_LOAD_IMAGE_GRAYSCALE)
	_, mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
	mask = mask[0:l_norm, 0:l_norm]

	print('Reading all images...')
	l_images = []
	images = []

	for f in fnames:
		prefix = dname + f[0:11]

		l1_fname = prefix + 'l1.jpg'
		l2_fname = prefix + 'l2.jpg'
		r1_fname = prefix + 'r1.jpg'
		r2_fname = prefix + 'r2.jpg'

		if f[0:11] == '8AD-visit1-':
			r2_fname = prefix + 'r2 (remedy).jpg'
		elif f[0:11] == 'MOI-visit2-':
			r2_fname = prefix + 'r2 (remedy).jpg'
		elif f[0:11] == 'UO2-visit2-':
			l2_fname = prefix + 'l2 (remedy).jpg'
			r2_fname = prefix + 'r2 (remedy).jpg'

		l1 = cv2.imread(l1_fname, cv2.CV_LOAD_IMAGE_COLOR)
		l2 = cv2.imread(l2_fname, cv2.CV_LOAD_IMAGE_COLOR)
		r1 = cv2.imread(r1_fname, cv2.CV_LOAD_IMAGE_COLOR)
		r2 = cv2.imread(r2_fname, cv2.CV_LOAD_IMAGE_COLOR)

		#-------------------------------------------------------------------------------
		l1 = l1[0:l_norm, 0:l_norm]
		l2 = l2[0:l_norm, 0:l_norm]
		r1 = r1[0:l_norm, 0:l_norm]
		r2 = r2[0:l_norm, 0:l_norm]
		#-------------------------------------------------------------------------------

		l_images.append(l1)
		l_images.append(l2)
		images.append(l1)
		images.append(l2)
		images.append(r1)
		images.append(r2)

	ncc_mat = []
	mi_mat = []

	print('Computing matching scores between each left ear and all others within the same visit...')
	for probe in l_images:
		ncc_v = [ compute_ncc(probe, gallery, mask) for gallery in images ]
		mi_v = [ compute_mi(probe, gallery) for gallery in images ]

		ncc_mat.append(ncc_v)
		mi_mat.append(mi_v)

	print('Output the matching scores...')
	csv_fname = '../../results/matching_ncc_v' + str(visit_num) + '.csv'
	try:
		remove(csv_fname)
	except OSError:
		pass
	csv_file = open(csv_fname, 'wb')
	csv_writer = writer(csv_file)
	csv_writer.writerows(ncc_mat)
	csv_file.close()

	csv_fname = '../../results/matching_mi_v' + str(visit_num) + '.csv'
	try:
		remove(csv_fname)
	except OSError:
		pass
	csv_file = open(csv_fname, 'wb')
	csv_writer = writer(csv_file)
	csv_writer.writerows(mi_mat)
	csv_file.close()
