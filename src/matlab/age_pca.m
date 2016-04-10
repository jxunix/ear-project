%
% Filename: age_pca.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Thu 07 Apr 2016 05:07:09 PM EDT
% Description: This script is to check the growth patterns using PCA.
%

clear; clc; close all;
load('../../results/symmetry_test_ws.mat', 'M', 'names');

%% clean the predictor X and response y
[ rows, cols ] = size(M);
X = reshape(M', cols*2, rows/2)';
[ rows, cols ] = size(X);

y = zeros(rows, 1);
id = 1;
for i = 1:rows
	if i > 1 & strcmp(names{i}(1:3), names{i-1}(1:3)) == 0
		id = id+1;
	end
	y(i) = id;
end

X_v3l1_idx = ~cellfun('isempty', strfind(names, '-visit3-l1'));
X_v3l2_idx = find(X_v3l1_idx) + 1;
X_v3r1_idx = X_v3l2_idx + 1;
X_v3r2_idx = X_v3l2_idx + 2;

X_v1l1_idx = X_v3l2_idx - 9;
X_v1l2_idx = X_v3l2_idx - 8;
X_v1r1_idx = X_v3l2_idx - 7;
X_v1r2_idx = X_v3l2_idx - 6;

X_v2l1_idx = X_v3l2_idx - 5;
X_v2l2_idx = X_v3l2_idx - 4;
X_v2r1_idx = X_v3l2_idx - 3;
X_v2r2_idx = X_v3l2_idx - 2;

X_v1l1 = X(X_v1l1_idx, :);
X_v1l2 = X(X_v1l2_idx, :);
X_v1r1 = X(X_v1r1_idx, :);
X_v1r2 = X(X_v1r2_idx, :);

X_v2l1 = X(X_v2l1_idx, :);
X_v2l2 = X(X_v2l2_idx, :);
X_v2r1 = X(X_v2r1_idx, :);
X_v2r2 = X(X_v2r2_idx, :);

X_v3l1 = X(X_v3l1_idx, :);
X_v3l2 = X(X_v3l2_idx, :);
X_v3r1 = X(X_v3r1_idx, :);
X_v3r2 = X(X_v3r2_idx, :);

X_v1l1_mean = mean(X_v1l1);
X_v1l2_mean = mean(X_v1l2);
X_v1r1_mean = mean(X_v1r1);
X_v1r2_mean = mean(X_v1r2);

X_v2l1_mean = mean(X_v2l1);
X_v2l2_mean = mean(X_v2l2);
X_v2r1_mean = mean(X_v2r1);
X_v2r2_mean = mean(X_v2r2);

X_v3l1_mean = mean(X_v3l1);
X_v3l2_mean = mean(X_v3l2);
X_v3r1_mean = mean(X_v3r1);
X_v3r2_mean = mean(X_v3r2);

Z = [ X_v1l1_mean; X_v1l2_mean; X_v1r1_mean; X_v1r2_mean; ...
			X_v2l1_mean; X_v2l2_mean; X_v2r1_mean; X_v2r2_mean; ...
			X_v3l1_mean; X_v3l2_mean; X_v3r1_mean; X_v3r2_mean ];

[ coeff, ~, latent, ~, explained ] = pca(Z);

Z_mean = mean(Z);
Z_centered = Z - repmat(Z_mean, 12, 1);
S = Z_centered' * Z_centered / (12 - 1);
[ U D V ] = svd(S);

idx = 1;
%% animation
a = 0;
b = 0.1;
while 1
	if a >= 3
		b = -b;
	elseif a <= -3
		b = -b;
	end
	a = a + b;
	Z_t = Z_mean + a * sqrt(D(idx, idx)) * coeff(:, idx)';
	scatter(Z_t(1:99), -Z_t(100:198), 'filled');
	xlim([300 1600]);
	ylim([-2600 -600]);
	pause(0.01);
end
