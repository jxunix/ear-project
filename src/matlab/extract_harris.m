%
% Filename: extract_harris.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Mon 04 Apr 2016 09:17:26 PM EDT
% Description: This script is to extract features (corners) using Harris–Stephens algorithm
%

dname = '../../outputs/4_flipped/';
fnames = dir(dname);
fnames = {fnames(3:end).name}';

%rng(0, 'twister');
%r = randi([1 372], 1,1);

fname = fnames{1};
fname = strcat(dname, fname);

I = imread(fname);
I = rgb2gray(I);
fprintf('%s\n', fname)
%image(I)

points = detectHarrisFeatures(I);
