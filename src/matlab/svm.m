%
% Filename: svm.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Sat 26 Mar 2016 07:36:50 PM EDT
% Description: This script is to classify the feature vectors by applying
%		Support Vector Machine.
%

% use libsvm package
addpath(fullfile(pwd, 'libsvm'));

load('../../results/symmetry_test_ws_manual.mat', 'M', 'names');

% clean the predictor X and response y
[ rows, cols ] = size(M);
X = reshape(M', cols*2, rows/2)';
[ rows, cols ] = size(X);

y_label = zeros(rows, 1);
id = 1;
for i = 1:rows
	if i > 1 & strcmp(names{i}(1:3), names{i-1}(1:3)) == 0
		id = id+1;
	end
	y_label(i) = id;
end

X_min = min(X(:));
X_max = max(X(:));
X = (X - X_min) / (X_max - X_min);
y = y_label;
%y = repmat(y_label,1,id) == repmat(1:id, rows, 1);
%y = 2*y-1;

% randomize
rindex = randperm(rows);
X = X(rindex, :);
y = y(rindex, :);

% 10-fold cross validation for SVM
num_folds = 10;
[ trainfolds, testfolds ] = kfold(rows, num_folds);

[ log2c, log2g ] = meshgrid(-10:10, -10:10);
train_acc = zeros(numel(log2c), num_folds);
test_acc = zeros(numel(log2c), num_folds);

for fold = 1:num_folds
	%fold = 1;
	train_id = trainfolds{fold};
	test_id = testfolds{fold};

	X_train = X(train_id, :);
	y_train = y(train_id);
	X_test = X(test_id, :);
	y_test = y(test_id);

	for i = 1:numel(log2c)
		% train SVM with RBF kernel
		% -s 0: learn classification SVM
		% -t 2: use Gaussian kernel
		% -c  : set the parameter C for cost
		% -g  : specify bandwidth
		% -q  : suppress console output
		libsvm_options = sprintf('-s 0 -t 2 -c %g -g %g -q', 2^log2c(i), 2^log2g(i));
		model = svmtrain(y_train, X_train, libsvm_options);

		[ y_pred_tr, ~, ~ ] = svmpredict(y_train, X_train, model);
		train_acc(i, fold) = 100 * mean(y_pred_tr == y_train);

		[ y_pred_tt, ~, ~ ] = svmpredict(y_test, X_test, model);
		test_acc(i, fold) = 100 * mean(y_pred_tt == y_test);
	end
end

[ val, idx ] = max(mean(train_acc, 2));
mean_test_acc = mean(test_acc, 2);

svm_train = val;
svm_test = mean_test_acc(idx);

fprintf('Best training accuracy: %0.1g%% when C=2^%d, gamma=2^%d.\n', val, log2c(idx), log2g(idx));
fprintf('The corresponding test accuracy: %0.1g%%.\n', svm_test);

train_std = std(train_acc, 0, 2);
test_std = std(test_acc, 0, 2);
svm_train_std = train_std(idx);
svm_test_std = test_std(idx);

ws_fname = '../../results/svm_ws_manual.mat';
save(ws_fname);
