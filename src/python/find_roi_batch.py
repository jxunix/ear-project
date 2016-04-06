#
# Filename: find_roi_batch.py
# Author: Jun Xu
# Email: junx@cs.bu.edu
# Created Time: Fri 04 Dec 2015 12:07:11 PM EST
# Description: This script finds the ROI in all images by matching with all
#		transformed templates. It will output the unwarped and standardized ROI's
#		under "outputs/5_detected_template_matching" and 
#		"outputs/6_detected_template_matching_standardized", respectively.
#

#-------------------------------------------------------------------------------
# import
#-------------------------------------------------------------------------------

import csv
import cv2
import numpy
import image
import template

from csv import writer
from os import listdir
from os import makedirs
from os import remove
from os.path import exists
from os.path import isfile
from os.path import join

#-------------------------------------------------------------------------------
# constants
#-------------------------------------------------------------------------------

scale_size = 6
angle_size = 7

log_scale = numpy.linspace(-1.0, 1.0, scale_size)
scale = [ pow(2, s) for s in log_scale ]
angle_in = numpy.linspace(-45, 45, angle_size)
angle_off = numpy.linspace(-45, 45, angle_size)

img = cv2.imread('../../resources/template_norm.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
l,_ = img.shape

#-------------------------------------------------------------------------------
# main body
#-------------------------------------------------------------------------------

print('Listing all templates...')
tpl_dname = '../../results/template_trans/'
tpl_fnames = [ f for f in listdir(tpl_dname) if isfile(join(tpl_dname, f)) ]
tpl_fnames.sort()

print('Creating a template object array...')
tpls = [ template.Template(tpl_dname + f) for f in tpl_fnames ]

print('Listing all images...')
img_dname = '../../outputs/4_flipped/'
img_fnames = [ f for f in listdir(img_dname) if isfile(join(img_dname, f)) ]
img_fnames.sort()

print('Creating a image object array...')
imgs = [ image.Image(img_dname + f) for f in img_fnames ]

dirname = '../../outputs/5_detected_template_matching/'
if not exists(dirname):
	makedirs(dirname)

dirname_norm = '../../outputs/6_detected_template_matching_standardized/'
if not exists(dirname_norm):
	makedirs(dirname_norm)

parameters = []

for i,img in enumerate(imgs):
	print('Processing template matching for image %s...' % img_fnames[i])
	rets = [ img.find_match(t) for t in tpls ]
	nccs = [ ret['ncc'] for ret in rets ]
	locs = [ ret['loc'] for ret in rets ]

	max_score = max(nccs)
	index = [ j for j,s in enumerate(nccs) if s == max_score ]
	index = index[0]
	print('  Max matching score (ncc) is %0.4f' % max_score)

	print('  Looking for the best match...')
	tpl = tpls[index]
	ncc = nccs[index]
	loc = locs[index]
	roi = img.find_roi(tpl, loc)
	print('  Outputing the ROI...')
	cv2.imwrite(dirname + img_fnames[i], roi)

	index_ai = index / (angle_size * scale_size)
	index %= angle_size * scale_size
	index_ao = index / scale_size
	index_s = index % scale_size

	ai = angle_in[index_ai]
	ao = angle_off[index_ao]
	s = scale[index_s]
	p = [ img_fnames[i], ai, ao, s ]
	parameters.append(p)
	print('  Transformation parameters are ai=%d, ao=%d, s=%0.3f' % (ai, ao, s))

	roi_norm = img.find_roi_norm(tpl, loc, l, ai, ao, s)
	print('  Outputing the standardized ROI...')
	cv2.imwrite(dirname_norm + img_fnames[i], roi_norm)

csv_fname = '../../results/transform_params.csv'
try:
	remove(csv_fname)
except OSError:
	pass

csv_file = open(csv_fname, 'wb')
csv_writer = writer(csv_file)
csv_writer.writerows(parameters)
csv_file.close()
