%
% Filename: extract_sift.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Tue 05 Apr 2016 12:13:08 PM EDT
% Description: This script is to extract SIFT features from ear images.
%

clear; close all; clc;

run('../../../vlfeat-0.9.20/toolbox/vl_setup.m')
vl_version verbose

%load('../../results/symmetry_test_ws.mat', 'M', 'names');
load('../../results/symmetry_test_ws_manual.mat', 'M', 'names');
dname = '../../outputs/4_flipped/';
[ rows, cols ] = size(M);

D = zeros(rows / 2 * 128, cols);
for i = 1:length(names)
%i = 1;
	fname = names{i};
	fname = strcat(dname, fname(1:14), 'jpg');
	fprintf('Processing image %s.\n', fname)

	I = imread(fname);
	I = single(rgb2gray(I));

	coord = M(2*i-1:2*i, :);
	scale = repmat(32, 1, cols);
	angle = repmat(0, 1, cols);
	keypt = [ coord; scale; angle ];
	[ f,d ] = vl_sift(I, 'frames', keypt, 'orientations');
    
	dup_flag = zeros(1, size(f,2));
	dup_flag(1) = 1;
	for j = 2:size(f,2)
		if f(1,j) ~= f(1, j-1)
			dup_flag(j) = 1;
		end
	end
	D(128 * i - 127:128 * i,:) = d(:, logical(dup_flag));
end

X = reshape(M', cols*2, rows/2)';
D = reshape(D', cols * 128, rows/2)';
X = [ X D ];

save('../../results/extract_sift_ws.mat');
