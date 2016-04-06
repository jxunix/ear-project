#
# Filename: organize_data.py
# Author: Jun Xu
# Email: junx@cs.bu.edu
# Created Time: Wed 19 Aug 2015 01:04:49 PM EDT
#
# Description: This script is to rename the images and create a form called 'organized_form.csv'
#              including all useful information.
#
#              After renaming, the format of the filename is 'participant_id-visit[1234]-[lr][12].jpg'.
#              @participant_id: a string consisting of 3 uppercase characters that uniquely identifies a child.
#                               ------Column S or AG in Registration_Forms.csv, or Column AJ in Visit_Forms.csv
#              @visit[1234]: the visit number.
#                            -------------------------------------------------Column Y or AU in Visit_Forms.csv
#              @[lr]: a flag that indicates whether it is the image of the left ear or that of the right ear.
#              @[12]: the first or the second image copy of the same ear.
#
#              The form 'organized_form.csv' contains the following fields:
#              @id: a string that uniquely identifies each visit.
#                   ----------------------------------------------------------Column B or AD in Visit_Forms.csv
#              @age: the age of the child during each visit.
#                    ---------------------------------------------------------Column W or AK in Visit_Forms.csv
#              @visit#: the visit number.
#                       ------------------------------------------------------Column Y or AU in Visit_Forms.csv
#              @case id: a string that uniquely identifies a child.
#                        --------------------Column Y in Registration_Forms.csv, or Column S in Visit_Forms.csv
#              @participant_id: a string consisting of 3 uppercase characters that uniquely identifies a child.
#                               ------Column S or AG in Registration_Forms.csv, or Column AJ in Visit_Forms.csv
#              @name: the name of the child.
#                     -------------------------------------------------Column V or AK in Registration_Forms.csv
#              @race: the race of the child.
#                     --------------------------------------------------------Column W in Registraion_Forms.csv
#              @gender: the gender of the child.
#                       -----------------------------------------------------Column X in Registration_Forms.csv
# 
#							 Note: There are 4 images missing. They are
#              1) ********-Visit-q2-right_ear_2-thea-f5cbc475-b8c1-418e-9e5b-65aacf99cc6c.jpg
#              2) ********-Visit-q2-right_ear_2-thea-be5fdef3-a5d0-43e5-a007-8177eb1a8151.jpg
#              3) ********-Visit-q3-left_ear_2-thea-0de22574-7d04-4423-97a2-3c5ff4ad201f.jpg
#              4) ********-Visit-q2-right_ear_2-thea-0de22574-7d04-4423-97a2-3c5ff4ad201f.jpg
#              The names of the children in the filenames are hidden deliberately.
#

import csv
import os
import sys

from PIL import Image

fname_visit = '../../config/Visit_Forms.csv'
fname_reg = '../../config/Registration_Forms.csv'
fname_form = '../../results/organized_form.csv'
dname_in = '../../resources/0_raw/'
dname_out = '../../outputs/1_renamed/'

print('1. Creating a new directory called 1_renamed')
if not os.path.exists(dname_out):
	os.makedirs(dname_out)

print('2. Creating a form called organized_form.csv')
reg_info = {}
f_reg = open(fname_reg)
csvreader = csv.reader(f_reg, delimiter = ',')
header = next(csvreader)
for row in csvreader:
	reg_info[row[24]] = row
f_reg.close()

visit_info = {}
f_visit = open(fname_visit)
csvreader = csv.reader(f_visit, delimiter = ',')
header = next(csvreader)
for row in csvreader:
	reg_entry = reg_info[row[18]]
	visit_info[row[1]] = [row[1], row[22], row[24], row[18], reg_entry[18], reg_entry[21], reg_entry[22], reg_entry[23]]
f_visit.close()

form_out = open(fname_form, 'w')
field_names = ["id", "age", "visit#", "case id", "participant id", "name", "race", "gender"]
wr = csv.DictWriter(form_out, fieldnames = field_names)
wr.writeheader()
for key, val in visit_info.items():
	if (val[5] == 'Baby testing' or
			val[5] == 'Demo 1' or
			val[5] == 'DUMMY TWO' or
			val[5] == 'CHRIS G'):
		continue
	wr = csv.writer(form_out, quoting = csv.QUOTE_ALL)
	wr.writerow(val)
form_out.close()

print('3. Renaming the image files and save them in directory 1_renamed')
print('   There are 4 images missing (for more information, refer to the script header).')
print('   So you will end up with 4 exceptions.')
for key, val in visit_info.items():
	if (val[5] == 'Baby testing' or
			val[5] == 'DEMO 1' or
			val[5] == 'DUMMY TWO' or
			val[5] == 'CHRIS G'):
		continue

	# rename 1st left ear image
	fname_in = dname_in + val[5] + '-Visit-q3-left_ear_1-thea-' + val[0] + '.jpg'
	fname_out = dname_out + val[4] + '-visit' + val[2] + '-l1.jpg'
	try:
		Image.open(fname_in).save(fname_out)
	except IOError:
		print('     cannot convert ' + fname_in)

	# rename 2nd left ear image
	fname_in = dname_in + val[5] + '-Visit-q3-left_ear_2-thea-' + val[0] + '.jpg'
	fname_out = dname_out + val[4] + '-visit' + val[2] + '-l2.jpg'
	try:
		Image.open(fname_in).save(fname_out)
	except IOError:
		print('     cannot convert ' + fname_in)

	# rename 1st right ear image
	fname_in = dname_in + val[5] + '-Visit-q2-right_ear_1-thea-' + val[0] + '.jpg'
	fname_out = dname_out + val[4] + '-visit' + val[2] + '-r1.jpg'
	try:
		Image.open(fname_in).save(fname_out)
	except IOError:
		print('     cannot convert ' + fname_in)

	# rename 2nd right ear image
	fname_in = dname_in + val[5] + '-Visit-q2-right_ear_2-thea-' + val[0] + '.jpg'
	fname_out = dname_out + val[4] + '-visit' + val[2] + '-r2.jpg'
	try:
		Image.open(fname_in).save(fname_out)
	except IOError:
		print('     cannot convert ' + fname_in)

print('Done!')
