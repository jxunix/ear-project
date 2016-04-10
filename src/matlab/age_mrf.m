%
% Filename: age_mrf.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Fri 08 Apr 2016 10:26:15 AM EDT
% Description: This script is to check the growth patterns using Markov Random Fields.
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

