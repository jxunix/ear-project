#
# Filename: find_roi.py
# Author: Jun Xu
# Email: junx@cs.bu.edu
# Created Time: Fri 04 Dec 2015 12:07:11 PM EST
# Description: This script finds the ROI in a image that matches the templates
#		best.
#

#-------------------------------------------------------------------------------
# import
#-------------------------------------------------------------------------------

import csv
import cv2
import numpy
import image
import template

from os import listdir
from os import makedirs
from os.path import exists
from os.path import isfile
from os.path import join

#-------------------------------------------------------------------------------
# constants
#-------------------------------------------------------------------------------

scale_size = 6
angle_size = 7

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

print('Reading the image...')
img_fname = '../../outputs/4_flipped/EUD-visit2-r1.jpg'
img = image.Image(img_fname)

print('Performing template matching...')
rets = [ img.find_match(t) for t in tpls ]
nccs = [ ret['ncc'] for ret in rets ]
locs = [ ret['loc'] for ret in rets ]

max_score = max(nccs)
index = [ i for i,s in enumerate(nccs) if s == max_score ]
index = index[0]
print('Max matching score (ncc) is %0.4f' % max_score)

print('Showing best matching...')
tpl = tpls[index]
ncc = nccs[index]
loc = locs[index]
roi = img.find_roi(tpl, loc)

index_ai = index / (angle_size * scale_size)
index %= angle_size * scale_size
index_ao = index / scale_size
index_s = index % scale_size
print('Transformation parameters: ai=%d, ao=%d, s=%d' % (index_ai, index_ao,
	index_s))

log_scale = numpy.linspace(-1.0, 1.0, scale_size)
scale = [ pow(2, s) for s in log_scale ]
angle_in = numpy.linspace(-45, 45, angle_size)
angle_off = numpy.linspace(-45, 45, angle_size)

ai = angle_in[index_ai]
ao = angle_off[index_ao]
s = scale[index_s]

roi_norm = img.find_roi_norm(tpl, loc, l, ai, ao, s)

roi_fname = '../../outputs/5_detected_template_matching/' + img_fname[26:43]
roi_norm_fname = '../../outputs/6_detected_template_matching_standardized/' + img_fname[26:43]
cv2.imwrite(roi_fname, roi)
cv2.imwrite(roi_norm_fname, roi_norm)
