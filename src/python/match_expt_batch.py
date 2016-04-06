#
# Filename: match_expt_batch.py
# Author: Jun Xu
# Email: junx@cs.bu.edu
# Created Time: Thu 10 Sep 2015 12:20:44 PM EDT
# Description: This script plots the NCC's against translation, scaling, and
#		in-/off-plane rotation for all images in each visit. This is because we
#		would like to find the transformed template that matches the ear region
#		best. To achieve that, we would like to assess how sensitive the NCC is to
#		the transformation.
#
#		After running this script, there will be a folder called
#		"matching_research" under "results". It contains three sub-folders.
#		Among them, "contractional_rotational_resolution" contains the plots of
#		the NCC's against scaling and in-/off-plane rotation,
#		"position_resolution" contains the plots of the NCC against translation,
#		and "results" contains the result matrices of template matching.
#

import cv2
import numpy

from math import cos
from math import pi
from math import sin
from math import sqrt

from matplotlib import pyplot

from os import listdir
from os import makedirs
from os.path import exists
from os.path import isfile
from os.path import join

#-------------------------------------------------------------------------------
# template preprocessing
#-------------------------------------------------------------------------------
# read template and compute its shape
templ_fname = '../../resources/template.jpg'
templ = cv2.imread(templ_fname, cv2.CV_LOAD_IMAGE_COLOR)
templ = cv2.resize(templ, (0,0), fx=0.1, fy=0.1)
templ_rows, templ_cols, _ = templ.shape

# compute alpha channel (mask) by treating black as transparency
templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
_, alpha = cv2.threshold(templ_gray, 0, 255, cv2.THRESH_BINARY);

#-------------------------------------------------------------------------------
# initialize the output directories
#-------------------------------------------------------------------------------
# list all image filenames in the directory
in_dirname = '../../outputs/4_flipped/'
image_fnames = [ f for f in listdir(in_dirname) \
		if isfile(join(in_dirname, f)) ]
image_fnames.sort()

# create result directories
# matching result directory
out_dirname1 = '../../results/matching_research/results/'
if not exists(out_dirname1):
	makedirs(out_dirname1)

# position resolution plot directory
out_dirname2 = '../../results/matching_research/position_resolution/'
if not exists(out_dirname2):
	makedirs(out_dirname2)

# contractional and rotational resolution plot directory
out_dirname3 = '../../results/matching_research/contractional_rotational_resolution/'
if not exists(out_dirname3):
	makedirs(out_dirname3)

#-------------------------------------------------------------------------------
# position resolution preparation
#-------------------------------------------------------------------------------
# "T" means template and "I" means image, used throughout the script
# sum(T^2) is compute once, but used for each image
T_mean = cv2.mean(templ, alpha)
T = cv2.subtract(templ, T_mean, mask=alpha, dtype=cv2.CV_32S)
T2 = cv2.pow(T, 2)
T2_sum = cv2.sumElems(T2)

#-------------------------------------------------------------------------------
# contractional resolution preparation
#-------------------------------------------------------------------------------
# scaled templates and masks
log_scale = numpy.linspace(-1.0, 1.0, 11)
x_scale = [ pow(2, s) for s in log_scale ]

m = cv2.moments(alpha, True)
templ_center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))

templ_scale_v = [ cv2.resize(templ, (0,0), fx=s, fy=s) for s in x_scale ]
alpha_scale_v = [ cv2.resize(alpha, (0,0), fx=s, fy=s) for s in x_scale ]

#-------------------------------------------------------------------------------
# rotational resolution preparation
#-------------------------------------------------------------------------------
length = int(sqrt(pow(templ_rows, 2) + pow(templ_cols, 2)))
half_length = int(length / 2)

# put the template to a larger canvas so that after rotation
# the template is still within the canvas
templ_base = numpy.zeros((length, length, 3), numpy.uint8)
templ_base[
		half_length - int(templ_rows / 2):
		half_length + templ_rows - int(templ_rows / 2),
		half_length - int(templ_cols / 2):
		half_length + templ_cols - int(templ_cols / 2)
		] = templ

# in-plane rotated templates and masks
x_rotate_in = numpy.linspace(-180, 180, 13)

templ_rotate_in_v = []
alpha_rotate_in_v = []

for theta in x_rotate_in:
	# perform rotation
	M = cv2.getRotationMatrix2D((half_length, half_length), theta, 1)
	templ_rotate = cv2.warpAffine(templ_base, M, (length, length))

	templ_rotate_gray = cv2.cvtColor(templ_rotate, cv2.COLOR_BGR2GRAY)
	_, alpha_rotate = cv2.threshold(templ_rotate_gray, 0, 255, cv2.THRESH_BINARY);

	templ_rotate_in_v.append(templ_rotate)
	alpha_rotate_in_v.append(alpha_rotate)

# off-plane rotate templates and masks
x_rotate_off = numpy.linspace(-60, 60, 13)

# parameters for off-plane rotation
# cited from http://jepsonsblog.blogspot.com/2012/11/rotation-in-3d-using-opencvs.html
f = 1000
dx = 0;
dy = 0;
dz = f
h, w, _ = templ_base.shape

A1 = numpy.array([
	[ 1, 0, -w / 2 ],
	[ 0, 1, -h / 2 ],
	[ 0, 0, 0 ],
	[ 0, 0, 1 ]
	])
Tr = numpy.array([
	[ 1, 0, 0, dx ],
	[ 0, 1, 0, dy ],
	[ 0, 0, 1, dz ],
	[ 0, 0, 0, 1 ]
	])
A2 = numpy.array([
	[ f, 0, w / 2, 0 ],
	[ 0, f, h / 2, 0 ],
	[ 0, 0, 1, 0 ]
	])

templ_off_rotate_v = []
alpha_off_rotate_v = []

for theta in x_rotate_off:
	theta *= pi / 180.0

	# perform off-plane rotation
	R = numpy.array([
		[ cos(theta), 0, -sin(theta), 0 ],
		[ 0, 1, 0, 0 ],
		[ sin(theta),  0, cos(theta), 0 ],
		[ 0, 0, 0, 1 ]
		])

	M = numpy.dot(A2, numpy.dot(Tr, numpy.dot(R, A1)))
	templ_rotate = cv2.warpPerspective(templ_base, M, (length, length))

	templ_rotate_gray = cv2.cvtColor(templ_rotate, cv2.COLOR_BGR2GRAY)
	_, alpha_rotate = cv2.threshold(templ_rotate_gray, 0, 255, cv2.THRESH_BINARY);

	templ_off_rotate_v.append(templ_rotate)
	alpha_off_rotate_v.append(alpha_rotate)

# for each image, compute the matching result and plots of NCC vs various transformations
for f in image_fnames:
	print('processing ' + f + '...')

	# read image and compute its shape
	image_fname = in_dirname + f
	image = cv2.imread(image_fname, cv2.CV_LOAD_IMAGE_COLOR)
	image = cv2.resize(image, (0,0), fx=0.1, fy=0.1)
	image_rows, image_cols, _ = image.shape

	# compute the height and width of result matrix
	# and initialize the result matrix
	result_rows = image_rows - templ_rows + 1
	result_cols = image_cols - templ_cols + 1
	result = numpy.zeros((result_rows, result_cols), dtype=float)

	#-----------------------------------------------------------------------------
	# position resolution calculation
	#-----------------------------------------------------------------------------
	# compute the NCC's
	for i in range(result_rows):
		for j in range(result_cols):
			I = image[i:i+templ_rows, j:j+templ_cols]

			I_mean = cv2.mean(I, alpha)
			I = cv2.subtract(I, I_mean, mask=alpha, dtype=cv2.CV_32S)
			I2 = cv2.pow(I, 2)
			I2_sum = cv2.sumElems(I2)

			IT = cv2.multiply(I, T)
			IT_sum = cv2.sumElems(IT)

			result[i,j] = sum(IT_sum) / sqrt(sum(I2_sum) * sum(T2_sum))

	# find the best matching result
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
	bottom_right = (max_loc[0] + templ.shape[1], max_loc[1] + templ.shape[0])
	cv2.rectangle(image, max_loc, bottom_right, 255, 2)

	# normalized correlation coefficients vs x/y coordinates
	x_htrans = range(result_cols)
	y_htrans = result[max_loc[1],:]

	x_vtrans = range(result_rows)
	y_vtrans = result[:,max_loc[0]]

	# output the matching results
	fig1, axarr1 = pyplot.subplots(ncols=3)

	axarr1[0].imshow(cv2.cvtColor(templ, cv2.COLOR_BGR2RGB))
	axarr1[0].set_title('Template')
	axarr1[0].axis('off')

	axarr1[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	axarr1[1].set_title('Detected Point')
	axarr1[1].axis('off')

	axarr1[2].imshow(result, cmap = 'gray')
	axarr1[2].set_title('Matching Result')

	fig1.savefig(out_dirname1 + 'general-' + f[0:13] + '.png')
	pyplot.close(fig1)

	# output the plot of NCC vs x/y coordinate
	fig2, axarr2 = pyplot.subplots(ncols=2, figsize=(18,12))

	axarr2[0].plot(x_htrans, y_htrans)
	axarr2[0].set_title('Normalized Correlation\nCoefficient vs X Coordinate')
	axarr2[0].set_xlabel('X coordinate')
	axarr2[0].set_ylabel('Normalized correlation coefficient')

	axarr2[1].plot(x_vtrans, y_vtrans)
	axarr2[1].set_title('Normalized Correlation\nCoefficient vs Y Coordinate')
	axarr2[1].set_xlabel('Y coordinate')
	axarr2[1].set_ylabel('Normalized correlation coefficient')

	fig2.savefig(out_dirname2 + 'general-' + f[0:13] + '.png')
	pyplot.close(fig2)

	#-----------------------------------------------------------------------------
	# contractional resolution calculation
	#-----------------------------------------------------------------------------
	image_center = [ sum(x) for x in zip(templ_center, max_loc) ]
	y_scale = []

	# compute the NCC's
	for i, templ_scale in enumerate(templ_scale_v):
		alpha_scale = alpha_scale_v[i]

		templ_scale_rows, templ_scale_cols, _ = templ_scale.shape

		templ_scale_rows_half = int(templ_scale_rows / 2)
		templ_scale_cols_half = int(templ_scale_cols / 2)

		# compute the xlimit and ylimit of the template.
		r_min = max(0, templ_scale_rows_half - image_center[1])
		c_min = max(0, templ_scale_cols_half - image_center[0])
		r_max = min(templ_scale_rows, templ_scale_rows_half + image_rows - image_center[1])
		c_max = min(templ_scale_cols, templ_scale_cols_half + image_cols - image_center[0])

		templ_scale = templ_scale[r_min:r_max, c_min:c_max]
		alpha_scale = alpha_scale[r_min:r_max, c_min:c_max]

		T_scale_mean = cv2.mean(templ_scale, alpha_scale)
		T_scale = cv2.subtract(templ_scale, T_scale_mean, mask=alpha_scale, dtype=cv2.CV_32S)
		T_scale2 = cv2.pow(T_scale, 2)
		T_scale2_sum = cv2.sumElems(T_scale2)

		# compute the xlimit and ylimit of ROI
		r_min = max(0, image_center[1] - templ_scale_rows_half)
		c_min = max(0, image_center[0] - templ_scale_cols_half)
		r_max = min(image_rows, image_center[1] + templ_scale_rows - templ_scale_rows_half)
		c_max = min(image_cols, image_center[0] + templ_scale_cols - templ_scale_cols_half)

		I_scale = image[r_min:r_max, c_min:c_max]

		I_scale_mean = cv2.mean(I_scale, alpha_scale)
		I_scale = cv2.subtract(I_scale, I_scale_mean, mask=alpha_scale, dtype=cv2.CV_32S)
		I_scale2 = cv2.pow(I_scale, 2)
		I_scale2_sum = cv2.sumElems(I_scale2)

		IT_scale = cv2.multiply(I_scale, T_scale)
		IT_scale_sum = cv2.sumElems(IT_scale)

		y_scale.append(sum(IT_scale_sum) / sqrt(sum(I_scale2_sum) * sum(T_scale2_sum)))

	#-----------------------------------------------------------------------------
	# in-plane rotation resolution calculation
	#-----------------------------------------------------------------------------
	y_rotate_in = []

	# these limits are used for off-plane rotation also
	# compute the xlimit and ylimit of ROI, fixed
	r_min = max(0, image_center[1] - half_length)
	c_min = max(0, image_center[0] - half_length)
	r_max = min(image_rows, image_center[1] + length - half_length)
	c_max = min(image_cols, image_center[0] + length - half_length)

	I_rotate_base = image[r_min:r_max, c_min:c_max]

	# compute the xlimit and ylimit of the template, fixed
	r_min = max(0, half_length - image_center[1])
	c_min = max(0, half_length - image_center[0])
	r_max = min(length, half_length + image_rows - image_center[1])
	c_max = min(length, half_length + image_cols - image_center[0])

	for i, templ_rotate in enumerate(templ_rotate_in_v):
		alpha_rotate = alpha_rotate_in_v[i]

		templ_rotate = templ_rotate[r_min:r_max, c_min:c_max]
		alpha_rotate = alpha_rotate[r_min:r_max, c_min:c_max]

		T_rotate_mean = cv2.mean(templ_rotate, alpha_rotate)
		T_rotate = cv2.subtract(templ_rotate, T_rotate_mean, mask=alpha_rotate, dtype=cv2.CV_32S)
		T_rotate2 = cv2.pow(T_rotate, 2)
		T_rotate2_sum = cv2.sumElems(T_rotate2)

		I_rotate_mean = cv2.mean(I_rotate_base, alpha_rotate)
		I_rotate = cv2.subtract(I_rotate_base, I_rotate_mean, mask=alpha_rotate, dtype=cv2.CV_32S)
		I_rotate2 = cv2.pow(I_rotate, 2)
		I_rotate2_sum = cv2.sumElems(I_rotate2)

		IT_rotate = cv2.multiply(I_rotate, T_rotate)
		IT_rotate_sum = cv2.sumElems(IT_rotate)

		y_rotate_in.append(sum(IT_rotate_sum) / sqrt(sum(I_rotate2_sum) * sum(T_rotate2_sum)))

	#-----------------------------------------------------------------------------
	# off-plane angular resolution calculation
	#-----------------------------------------------------------------------------
	y_rotate_off = []

	for i, templ_rotate in enumerate(templ_off_rotate_v):
		alpha_rotate = alpha_off_rotate_v[i]

		templ_rotate = templ_rotate[r_min:r_max, c_min:c_max]
		alpha_rotate = alpha_rotate[r_min:r_max, c_min:c_max]

		T_rotate_mean = cv2.mean(templ_rotate, alpha_rotate)
		T_rotate = cv2.subtract(templ_rotate, T_rotate_mean, mask=alpha_rotate, dtype=cv2.CV_32S)
		T_rotate2 = cv2.pow(T_rotate, 2)
		T_rotate2_sum = cv2.sumElems(T_rotate2)

		I_rotate_mean = cv2.mean(I_rotate_base, alpha_rotate)
		I_rotate = cv2.subtract(I_rotate_base, I_rotate_mean, mask=alpha_rotate, dtype=cv2.CV_32S)
		I_rotate2 = cv2.pow(I_rotate, 2)
		I_rotate2_sum = cv2.sumElems(I_rotate2)

		IT_rotate = cv2.multiply(I_rotate, T_rotate)
		IT_rotate_sum = cv2.sumElems(IT_rotate)

		y_rotate_off.append(sum(IT_rotate_sum) / sqrt(sum(I_rotate2_sum) * sum(T_rotate2_sum)))

	# output the plot the NCC vs contraction and in-/off-plane rotation
	fig3, axarr3 = pyplot.subplots(ncols=3, figsize=(18,12))

	axarr3[0].plot(log_scale, y_scale)
	axarr3[0].set_title('Normalized Correlation\nCoefficient vs Log2-Scale')
	axarr3[0].set_xticks(log_scale)
	axarr3[0].set_xlabel('Log2-scale')
	axarr3[0].set_ylabel('Normalized correlation coefficient')

	axarr3[1].plot(x_rotate_in, y_rotate_in)
	axarr3[1].set_title('Normalized Correlation\nCoefficient vs In-Plane Rotation Angle')
	axarr3[1].set_xticks(x_rotate_in)
	axarr3[1].set_xlabel('Angle in degree')
	axarr3[1].set_ylabel('Normalized correlation coefficient')

	axarr3[2].plot(x_rotate_off, y_rotate_off)
	axarr3[2].set_title('Normalized Correlation\nCoefficient vs Off-Plane Rotation Angle')
	axarr3[2].set_xticks(x_rotate_off)
	axarr3[2].set_xlabel('Angle in degree')
	axarr3[2].set_ylabel('Normalized correlation coefficient')

	fig3.savefig(out_dirname3 + 'general-' + f[0:13] + '.png')
	pyplot.close(fig3)
