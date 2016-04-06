%
% Filename: extract_sift.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Tue 05 Apr 2016 12:13:08 PM EDT
% Description: This script is to extract SIFT features from ear images.
%

run('../../../vlfeat-0.9.20/toolbox/vl_setup.m')
vl_version verbose

%%
crv_fname = '../../asm/models/ear_99pts.crvs';
crvs = textread(crv_fname, '%s', 'delimiter', '\n');
crv = crvs{2};
crv = strsplit(crv, ' ');
crv = crv(9:44);
polyline = zeros(length(crv), 1);
for i = 1:length(crv)
	polyline(i) = str2num(crv{i});
end

%%
pt_dname = '../../asm/points_99/';
pt_fnames = dir(pt_dname);
pt_fnames = {pt_fnames(3:end).name}';
xv = zeros(length(pt_fnames), length(polyline));
yv = zeros(length(pt_fnames), length(polyline));

for i = 1:length(pt_fnames)
	f = strcat(pt_dname, pt_fnames{i});
	pts = textread(f, '%s', 'delimiter', '\n');
	pts = pts(4:end-1);
	pts = pts(polyline + 1);
	for j = 1:length(polyline)
		pt = pts{j};
		pt = strsplit(pt, ' ');
		xv(i, j) = str2num(pt{1});
		yv(i, j) = str2num(pt{2});
	end
end

%%
dname = '../../outputs/4_flipped/';
fnames = dir(dname);
fnames = {fnames(3:end).name}';

% extract SIFT features from the images
descriptors = cell(length(fnames), 1);
for i = 1:length(fnames)
	fname = strcat(dname, fnames{i});
	fprintf('%d: %s\n', i, fname);
	I = imread(fname);
	%I = imresize(I, 0.1);
	I = single(rgb2gray(I));

	[ f d ] = vl_sift(I);
	in = inpolygon(f(1,:), f(2,:), xv(i,:), yv(i,:));
	d = d(:, in);
	descriptors{i} = { d };
end

%% compute pair-wise distances
dists = zeros(length(fnames));
num_pts = 50;

for i = 1:length(descriptors)
	for j = 1:length(descriptors)
	%i=1;j=2;
		fprintf('i=%d, j=%d: %s vs %s\n', i, j, fnames{i}, fnames{j});

		[ matches, scores ] = vl_ubcmatch(descriptors{i}, descriptors{j});
		scores = sort(scores, 'descend');
		dists(1, i) = sum(scores(1:num_pts), 2);
	end
end

save('./sift.mat');
