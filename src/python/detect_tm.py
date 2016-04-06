#
# Filename: detect_tm.py
# Author: Jun Xu
# Email: junx@cs.bu.edu
# Created Time: Wed 16 Mar 2016 09:11:30 PM EDT
# Description: This script is to detect ear regions by applying generalized
#		template matching. The input data are the images of those who had 3 visits.
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

print('Creating a image object array...')
imgs = [ image.Image(img_dname + f) for f in img_fnames ]

dirname = '../../outputs/detected_tm/'
if not exists(dirname):
	makedirs(dirname)

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
	roi = img.find_matching_boundary(tpl, loc)
	print('  Outputing the ear regions...')
	cv2.imwrite(dirname + img_fnames[i], roi)
	# cv2.imshow('test', roi)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
