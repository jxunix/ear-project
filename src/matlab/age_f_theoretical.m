%
% Filename: age_f_theoretical.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Sat 16 Apr 2016 08:27:25 PM EDT
% Description: This script is to compute the theoretical improvement of
%   recognition rate if we adjust for the growth pattern. We assume that
%   X_i = f(X_j) for all j != i.
%

clear; close all; clc;
load('../../results/symmetry_test_ws.mat', 'M', 'names');

%% clean the predictor X and response class
[ rows, cols ] = size(M);
X = reshape(M', cols*2, rows/2)';
[ rows, cols ] = size(X);

class = zeros(rows, 1);
id = 1;
for i = 1:rows
	if i > 1 & strcmp(names{i}(1:3), names{i-1}(1:3)) == 0
		id = id+1;
	end
	class(i) = id;
end

X1_idx = ~cellfun('isempty', strfind(names, '-visit1-'));
X2_idx = ~cellfun('isempty', strfind(names, '-visit2-'));
X3_idx = ~cellfun('isempty', strfind(names, '-visit3-'));

X1_idx = find(X1_idx);
X2_idx = find(X2_idx);
X3_idx = find(X3_idx);

X1 = X(X1_idx, :);
X2 = X(X2_idx, :);
X3 = X(X3_idx, :);

X1_upto2_idx = X2_idx - 4;
X1_upto3_idx = X3_idx - 8;
X2_upto3_idx = X3_idx - 4;

X1_upto2 = X(X1_upto2, :);
X1_upto3 = X(X1_upto3, :);
X2_upto3 = X(X2_upto3, :);

%% machine learning model 1
%  to predict ear shape at the 3rd visit for those tracked up to the 2nd visit
X_train = [ X1_upto3; X2_upto3 ];
y_train = X3;
X_test = [ X1_upto2; X2 ]; 
y_train_pred = 
y_test_pred = ?;

%% machine learning model 2
%  to classify the ear shape and find newborns' ID
X_train = ;
y_train = ;
