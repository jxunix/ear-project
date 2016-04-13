%
% Filename: age_kalman.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Fri 08 Apr 2016 10:26:15 AM EDT
% Description: This script is to check the growth patterns using Kalman Filter.
%   At first, we tried to model the problem by Hidden Markov Model. But the states
%   of HMM are discrete. When it comes to continuous states, it's called Linear
%   Dynamic System, whose solution could be Kalman Filter.
%

clear; clc; close all;
load('../../results/symmetry_test_ws_manual.mat', 'M', 'names');

%% clean the predictor X and response y
[ rows, cols ] = size(M);
X = reshape(M', cols*2, rows/2);
[ rows, cols ] = size(X);
X_mean = mean(X, 2);

X3_idx = ~cellfun('isempty', strfind(names, '-visit3-'));
X2_idx = find(X3_idx) - 4;
X1_idx = X2_idx - 4;

X1 = X(:, X1_idx);
X2 = X(:, X2_idx);
X3 = X(:, X3_idx);

i = 1;
Xi1 = X1(:, 4*i-3:4*i);
Xi2 = X2(:, 4*i-3:4*i);
Xi3 = X3(:, 4*i-3:4*i);

Xi1_mean = mean(Xi1, 2);
Xi2_mean = mean(Xi2, 2);
Xi3_mean = mean(Xi3, 2);
Xi = [ Xi1_mean Xi2_mean Xi3_mean ];
Xi_mean = mean([ Xi1 Xi2 Xi3 ], 2);

%%
A = ;
H = eye(rows);
Q = eye(rows) * (10 ^ -2);
R = diag(var(Xi - repmat(Xi_mean, 1, 3), 0, 2));
Pk_1 = 0.1;
Xk_1 = X_mean;

for j = 1:3
	Xk = A * Xk_1;
	Pk = A * Pk_1 * A' + Q;

	Kk = (H * Pk * H' + R) \ (Pk * H');
	Xk = Xk + Kk * (Xi(j,:) - H * Xk);
	Pk = (eye(rows) - Kk * H) * Pk;

	Xk_1 = Xk;
	Pk_1 = Pk;

	Xk_track(j,:) = Xk;
end
