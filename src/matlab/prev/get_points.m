%
% Filename: get_points.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Fri 18 Dec 2015 01:32:22 PM EST
% Description: This script records the mouse click coordinates in the image. The
%		coordinates are output to a text file.
%

clear;
close all;

fname = '../../../resources/visit3/3CI-visit3-l1.jpg'
base_name = fname(1:37);

fnames = { fname [base_name, 'l2.jpg'] [base_name, 'r1.jpg'] [base_name, 'r2.jpg'] };

for fname=fnames
	image = imread(fname);
	image = imresize(image, 0.25);
	imshow(image);

	points = [];
	n = 10;

	hold on
	for i=1:n
		point = ginput(1);
		points = [ points; point ];
		plot(points(:,1), points(:,2), 'r+');
	end
	hold off

	points = points ./ 2.5;

	outDname = [ '../../../results/visit', fname(25:25), '/point_loc/', fname(27:39), '.txt' ];
	dlmwrite(outDname, points);
end
