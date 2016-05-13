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

X1_upto2 = X(X1_upto2_idx, :);
X1_upto3 = X(X1_upto3_idx, :);
X2_upto3 = X(X2_upto3_idx, :);

%% machine learning model 1
%  to predict ear shape at the 3rd visit for those tracked up to the 2nd visit
X_train = [ repmat(1, length(X3_idx), 1) X1_upto3 X2_upto3 ];
y_train = X3;
X_test = [ repmat(1, length(X2_idx), 1) X1_upto2 X2 ]; 

omega = pinv(X_train) * y_train;
y_test_pred = X_test * omega;

%% machine learning model 2
%  to classify the ear shape and find newborns' ID
X_train_treat = [ X1_upto2; X2; y_test_pred ];
y_train_treat = [ class(X2_idx); class(X2_idx); class(X2_idx) ];

X_train_ctrl = [ X1_upto2; X2 ];
y_train_ctrl = [ class(X2_idx); class(X2_idx) ];

X_test = X3;
y_test = class(X3_idx);

d_treat = pdist2(X_train_treat, X_test);
d_ctrl = pdist2(X_train_ctrl, X_test);

Ks = 1:15;
acc_treat = zeros(length(Ks), 1);
acc_ctrl = zeros(length(Ks), 1);

for K = 1:length(Ks)
	y_pred_treat = sol_knn_classify(X_train_treat, y_train_treat, X_test, K, d_treat);
	y_pred_ctrl = sol_knn_classify(X_train_ctrl, y_train_ctrl, X_test, K, d_ctrl);

	acc_treat(K) = 100 * mean(y_pred_treat == y_test);
	acc_ctrl(K) = 100 * mean(y_pred_ctrl == y_test);
end

% plot cross validation accuracy against K
plot(Ks, acc_treat, '-ok');
hold on
plot(Ks, acc_ctrl, '--o', 'color', [ 0.5, 0.5, 0.5 ]);
plot(xlim, [ 1/31 1/31 ], '--k')
hold off
xlabel('K')
ylabel('Recognition rate %')
ylim([0 100])
%title('Accuracy vs. K');
legend('Adjusted', 'Unadjusted');

%%
outname = '../../results/theoretical_improvement_adjusted2.png';
print(outname, '-dpng')

[ val, idx ] = max(acc_treat);
fprintf('Best accuracy w/ correcting for growth is %0.1f%% when K=%d.\n', val, idx)
[ val_ctrl, idx_ctrl ] = max(acc_ctrl);
fprintf('Best accuracy w/o correcting for growth is %0.1f%% when K=%d.\n', val_ctrl, idx_ctrl)
mean(acc_treat - acc_ctrl)
