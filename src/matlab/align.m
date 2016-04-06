%
% Filename: align.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Sat 19 Mar 2016 10:05:51 AM EDT
% Description: This script is to align all the feature vectors through
%   Procrustes Analysis.
%

function M_norm = align(M)

%fname = '../../results/features_manual.csv';
%M = csvread(fname);

d = size(M);
rows = d(2);
cols = 2;
pages = d(1) / cols;
M_3d = reshape(M', rows, cols, pages);

X0 = mean(M_3d, 3);
scale = sqrt(sum(sum(X0 .^ 2), 2));

numIter = 200;
threshold = 1;

for i = 1:numIter
	disp(['Iteration ', int2str(i)])
	for j = 1:pages
		[ d,Z ] = procrustes(X0, M_3d(:,:,j), 'scaling', false);
		M_3d(:,:,j) = Z;
	end

	X00 = mean(M_3d, 3);
	s = sqrt(sum(sum(X00 .^ 2), 2));
	X00 = X00 .* (scale / s);
	diff = (X00 - X0) .^ 2;
	diff = sqrt(sum(sum(diff), 2));

	disp(['  Difference: ', num2str(diff)])

	if diff < threshold
		disp('  Converged')
		break
	end

	X0 = X00;
end

M_norm = reshape(M_3d, rows, pages*2);
M_norm = M_norm';
