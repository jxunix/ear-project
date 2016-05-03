%
% Filename: age_lm_theoretical.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Thu 14 Apr 2016 12:43:31 PM EDT
% Description: This script is to compute the theoretical improvement of
%   recognition rate if we adjust for the growth pattern. We model the
%   growth pattern through linear regression, in particular, X_i|X_{i-1} ~
%   Normal(A * X_{i-1} + b, \Sigma).
%

clear; close all; clc;
load('../../results/symmetry_test_ws_asm.mat', 'M', 'names');

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

X3_idx = ~cellfun('isempty', strfind(names, '-visit3-'));
X3_idx = find(X3_idx);
X2_idx = X3_idx - 4;
X1_idx = X2_idx - 4;

X2_idx2 = ~cellfun('isempty', strfind(names, '-visit2-'));
X2_idx2 = find(X2_idx2);
X1_idx2 = X2_idx2 - 4;

X3 = X(X3_idx, :);
X2 = X(X2_idx, :);
X1 = X(X1_idx, :);
X2_2 = X(X2_idx2, :);
X1_2 = X(X1_idx2, :);

y = [ X3; X2_2 ];
Z = [ X2; X1_2 ];
Z = [ repmat(1, size(Z,1), 1) Z ];

%% linear regression when it's under-determined
w = pinv(Z) * y;

%% display
crv_fname = '../../asm/models/ear_99pts.crvs';
crvs = textread(crv_fname, '%s', 'delimiter', '\n');

crv1 = strsplit(crvs{2}, ' ');
crv2 = strsplit(crvs{3}, ' ');
crv3 = strsplit(crvs{4}, ' ');

crv1 = crv1(9:end-3);
crv2 = crv2(9:end-3);
crv3 = crv3(9:end-3);

c1 = zeros(1, length(crv1));
c2 = zeros(1, length(crv2));
c3 = zeros(1, length(crv3));

for i = 1:length(crv1)
	c1(i) = str2num(crv1{i}) + 1;
end
for i = 1:length(crv2)
	c2(i) = str2num(crv2{i}) + 1;
end
for i = 1:length(crv3)
	c3(i) = str2num(crv3{i}) + 1;
end

ID = names(X3_idx);

%%
X3_pred = [ repmat(1, size(X2_2, 1), 1) X2_2 ] * w;

X1_idx3 = ~cellfun('isempty', strfind(names, '-visit1-'));
X1_3 = X(X1_idx3, :);
X_train = [ X1_3; X2_2; X3_pred ];
X_train_ctrl = [ X1_3; X2_2 ];
X_test = X3;

y1 = class(X1_idx3);
y2 = class(X2_idx2);
y3 = class(X3_idx);
y_train = [ y1; y2; y2 ];
y_train_ctrl = [ y1; y2 ];
y_test = y3;

d = pdist2(X_train, X_test);
d_ctrl = pdist2(X_train_ctrl, X_test);
Ks = 1:5:size(X_train_ctrl) / 2;
acc = zeros(length(Ks), 1);
acc_ctrl = zeros(length(Ks), 1);

for K = 1:length(Ks)
	y_pred = sol_knn_classify(X_train, y_train, X_test, K, d);
	y_pred_ctrl = sol_knn_classify(X_train_ctrl, y_train_ctrl, X_test, K, d_ctrl);

	acc(K) = 100 * mean(y_pred == y_test);
	acc_ctrl(K) = 100 * mean(y_pred_ctrl == y_test);
end

% plot cross validation accuracy against K
plot(Ks, acc, '-ok');
hold on
plot(Ks, acc_ctrl, '--o', 'color', [ 0.5, 0.5, 0.5 ]);
hold off
xlabel('K')
ylabel('Accuracy %')
title('Accuracy vs. K');
legend('Adjusted', 'Unadjusted');

outname = '../../results/theoretical_improvement_adjusted.png';
print(outname, '-dpng')

[ val, idx ] = max(acc);
fprintf('Best accuracy w/ correcting for growth is %0.1f%% when K=%d.\n', val, idx)
[ val_ctrl, idx_ctrl ] = max(acc_ctrl);
fprintf('Best accuracy w/o correcting for growth is %0.1f%% when K=%d.\n', val_ctrl, idx_ctrl)
