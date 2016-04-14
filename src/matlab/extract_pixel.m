%
% Filename: extract_pixel.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Thu 14 Apr 2016 06:00:06 PM EDT
% Description: This script is to extract the pixel (rgb) values of all feature points.
%

clear; close all; clc;

%% read the image filenames
fname = '../../results/image_names.txt';
names = textread(fname, '%s', 'delimiter', '\n');

%% read the coordinates of the feature points and remove missing data (whose
%  corresponding filename contains '(remedy)'.
%fname = '../../results/features_asm.csv';
fname = '../../results/features_manual.csv';
M = csvread(fname);

rows = size(M,1);
index = zeros(rows, 1);
for i = 1:rows
	s = size(strfind(names{ceil(i/2)}, 'remedy'), 2);
	
	if s == 1
		i = ceil(i/8)*8+1;
		index(i-8) = 1;
		index(i-7) = 1;
		index(i-6) = 1;
		index(i-5) = 1;
		index(i-4) = 1;
		index(i-3) = 1;
		index(i-2) = 1;
		index(i-1) = 1;
	end
end

index = logical(index);
M = M(~index, :);
[ rows, cols ] = size(M);

index = index(1:2:length(index));
names = names(~index);

dname = '../../outputs/4_flipped/';
rgbs = zeros(rows/2, cols*3);

for i = 1:length(names)
%i = 1;
	fname = names{i};
	fname = strcat(dname, fname(1:14), 'jpg');
	fprintf('Processing image %s\n', fname)

	I = imread(fname);
	r = M(2*i-1, :);
	c = M(2*i, :);
	pixels = impixel(I,r,c);
	rgbs(i,:) = pixels(:)';
end

ws_fname = '../../results/extract_pixel_ws.mat';
save(ws_fname);
