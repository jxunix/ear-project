%
% Filename: svm_interface.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Tue 29 Mar 2016 09:36:10 PM EDT
% Description: This script is to classify features by calling multi-class SVM.
%

%% preprocessing
load('../../results/symmetry_test_ws_manual.mat', 'M', 'names');

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

train_errs = zeros(length(lambdas) + 1, num_folds);
test_errs = zeros(length(lambdas) + 1, num_folds);

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

	[ y_pred, y_pred_debug, y_prob, train_err ] = svm_multi(X_train, y_train, X_test, label_train, id, rindex_train, rindex_test, 0);
	train_errs(1, fold) = train_err;
	test_errs(1, fold) = mean(y_pred ~= y_test);

	for i = 1:length(lambdas)
		lambda = lambdas(i);
		[ y_pred, y_pred_debug, y_prob, train_err ] = svm_multi(X_train, y_train, X_test, label_train, id, rindex_train, rindex_test, 1, lambda);
		train_errs(i+1, fold) = train_err;
		test_errs(i+1, fold) = mean(y_pred ~= y_test);
	end
end

%% draw the bar plot
test_errs = test_errs * 100;
test_accs = 100 - test_errs;
bar(mean(test_accs, 2), 'FaceColor', [ 0.5,0.5,0.5 ]);
set(gca, 'XTickLabel', { '0', '10^-6', '10^-5', '10^-4', '10^-3', '10^-2', '10^-1', '1', '10', '10^2' })
xlabel('\lambda')
ylabel('Recognition rate %')
%title('10-fold Cross Validation Error vs \lambda')
hold on
errorbar(mean(test_accs, 2), std(test_accs, 0, 2), '.k');
hold off

outname = '../../results/svm_penalized_error_manual.png';
print(outname, '-dpng')

train_accs = (1 - train_errs) * 100;

[ val, idx ] = max(mean(train_accs, 2));
mean_test_acc = mean(test_accs, 2);

svm_train = val;
svm_test = mean_test_acc(idx);

train_std = std(train_accs, 0, 2);
test_std = std(test_accs, 0, 2);
svm_train_std = train_std(idx);
svm_test_std = test_std(idx);

ws_fname = '../../results/svm_own_ws_manual.mat';
save(ws_fname);
