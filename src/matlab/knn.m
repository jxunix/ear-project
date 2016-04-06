%
% Filename: knn.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Tue 22 Mar 2016 05:09:35 PM EDT
% Description: This script is to classify the feature vectors by applying KNN.
%

load('../../results/symmetry_test_ws.mat', 'M', 'names');

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

% randomize
rindex = randperm(rows);
X = X(rindex, :);
y = y(rindex, :);

% precompute entire pairwise distance matrix
dists = pdist2(X, X);

% 10-fold cross validation for KNN
num_folds = 10;
[ trainfolds, testfolds ] = kfold(rows, num_folds);

Ks = 1:5:rows/2;
err = zeros(length(Ks), num_folds);
for K = 1:length(Ks)
	for fold = 1:num_folds
		fprintf('K=%d, fold=%d\n', Ks(K), fold);
		train_id = trainfolds{fold};
		test_id = testfolds{fold};

		X_train = X(train_id, :);
		y_train = y(train_id);
		X_test = X(test_id, :);
		y_test = y(test_id);
		ds = dists(train_id, test_id);

		y_pred = sol_knn_classify(X_train, y_train, X_test, K, ds);
		err(K, fold) = 100 * mean(y_pred ~= y_test);
	end
end

% plot cross validation accuracy against K
acc = 100 - err;
errorbar(Ks, mean(acc, 2), std(acc, 0, 2), '-ok');
xlabel('K')
ylabel('Accuracy %')
title(sprintf('%d-fold Cross Validation Accuracy vs. K', num_folds));

outname = '../../results/knn_acc.png';
print(outname, '-dpng')

[ val, idx ] = max(mean(acc, 2));
fprintf('Best accuracy is %0.1f%% when K=%d.\n', val, idx)

knn_test = val;
stdev = std(acc, 0, 2);
knn_test_std = stdev(idx);

ws_fname = '../../results/knn_ws.mat';
save(ws_fname);
