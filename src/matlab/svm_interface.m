%
% Filename: svm_interface.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Tue 29 Mar 2016 09:36:10 PM EDT
% Description: This script is to classify features by calling multi-class SVM.
%

%% preprocessing
load('work_space.mat', 'M', 'names');

% clean the predictor X and response y
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

label = zeros(rows, 1);
lb = 1;
for i = 1:rows
	if i > 1 & strcmp(names{i}(1:10), names{i-1}(1:10)) == 0
		lb = lb + 1;
	end
	label(i) = lb;
end

X_min = min(X(:));
X_max = max(X(:));
X = (X - X_min) / (X_max - X_min);

% randomize
rindex = randperm(rows);
X = X(rindex, :);
y = y(rindex, :);
label = label(rindex);

%% 10-fold cross validation for SVM
num_folds = 10;
[ trainfolds, testfolds ] = kfold(rows, num_folds);

log2lambdas = [-6:1:2];
lambdas = 10 .^ log2lambdas;

train_errs = zeros(num_folds, length(lambdas) + 1);
test_errs = zeros(num_folds, length(lambdas) + 1);

for fold = 1:num_folds
%fold = 1;
	fprintf('fold %d\n', fold)
	train_id = trainfolds{fold};
	test_id = testfolds{fold};

	X_train = X(train_id, :);
	y_train = y(train_id);
	rindex_train = rindex(train_id);
	label_train = label(train_id);

	X_test = X(test_id, :);
	y_test = y(test_id);
	rindex_test = rindex(test_id);

	[ y_pred, train_err ] = svm_multi(X_train, y_train, X_test, label_train, id, rindex_train, rindex_test, 0);
	train_errs(fold, 1) = train_err;
	test_errs(fold, 1) = mean(y_pred ~= y_test);

	for i = 1:length(lambdas)
		lambda = lambdas(i);
		[ y_pred, train_err ] = svm_multi(X_train, y_train, X_test, label_train, id, rindex_train, rindex_test, 1, lambda);
		train_errs(fold, i+1) = train_err;
		test_errs(fold, i+1) = mean(y_pred ~= y_test);
	end
end

save('./svm.mat');

%% draw the bar plot
test_errs = test_errs * 100;
bar(mean(test_errs), 'FaceColor', [ 0.5,0.5,0.5 ]);
set(gca, 'XTickLabel', { '0', '10^-6', '10^-5', '10^-4', '10^-3', '10^-2', '10^-1', '1', '10', '10^2' })
xlabel('\lambda')
ylabel('Test Error %')
title('10-fold Cross Validation Error vs \lambda')
hold on
errorbar(mean(test_errs), std(test_errs), '.k');
hold off

outname = '../../results/svm_penalized_error.png';
print(outname, '-dpng')

rr_fname = './recognition_rate.mat';
svm_train_own = 100 - min(mean(train_errs), [], 2);
svm_test_own = 100 - min(mean(test_errs), [], 2);
save(rr_fname, 'svm_train_own', '-append');
save(rr_fname, 'svm_test_own', '-append');
