%
% Filename: flda.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Wed 23 Mar 2016 08:59:42 PM EDT
% Description: This script is to classify the feature vectors by applying Fisher
%		Discriminant Linear Analysis.
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

% 10-fold cross validation for FLDA
num_folds = 10;
[ trainfolds, testfolds ] = kfold(rows, num_folds);

classes = unique(y)';
num_cls = length(classes);

train_err = zeros(num_cls-1, num_folds);
test_err = zeros(num_cls-1, num_folds);

% coefficient used for within-class variance matrix regularization
gamma = 0.05;

for fold = 1:num_folds
	train_id = trainfolds{fold};
	test_id = testfolds{fold};

	X_train = X(train_id, :);
	y_train = y(train_id);
	X_test = X(test_id, :);
	y_test = y(test_id);

	%% compute within-class variance and between-class variance
	Sw = zeros(cols);
	Sb = zeros(cols);
	mu = mean(X_train);
	mu_c = zeros(num_cls, cols);

	for i = 1:num_cls
		c = classes(i);
		idx = find(y_train == c);
		mu_c(i,:) = mean(X_train(idx, :));

		Sb = Sb + length(idx) * (mu_c(i,:) - mu)' * (mu_c(i,:) - mu);
		for j = idx'
			Sw = Sw + (X_train(j,:) - mu_c(i,:))' * (X_train(j,:) - mu_c(i,:));
		end
	end

	% compute eigenvectors of Sw^{-1}*Sb and sort according to eigenvalues
	% because within-class varinace matrix is singular, we use regularized FLDA
	% instead (https://www.stat.tamu.edu/~jianhua/paper/iccsde-sparseLDA.pdf)
	Sw = Sw + gamma / cols * trace(Sw) * eye(cols);
	[ V,D ] = eig(Sw \ Sb);
	[ ~, idx ] = sort(diag(D), 'descend');
	V = V(:, idx);

	%% classification with varying number of eigenvectors
	for n = 1:num_cls-1
		W = V(:, 1:n);

		centers_proj = mu_c * W;
		X_train_proj = X_train * W;
		X_test_proj = X_test * W;

		y_pred = sol_knn_classify(centers_proj, classes, X_train_proj, 1)';
		train_err(n, fold) = mean(y_pred ~= y_train);
		y_pred = sol_knn_classify(centers_proj, classes, X_test_proj, 1)';
		test_err(n, fold) = mean(y_pred ~= y_test);
	end
end

%% plot and report
train_acc = (1 - train_err) * 100;
test_acc = (1 - test_err) * 100;

figure(2)
errorbar(1:num_cls-1, mean(train_acc, 2), std(train_acc, 0, 2), '-ok');
hold on
errorbar(1:num_cls-1, mean(test_acc, 2), std(test_acc, 0, 2), '--k');
legend({'Training accuracy', 'Test accuracy'});
xlabel('Number of Eigenvectors');
ylabel('Accuracy %')
title(sprintf('%d-fold Cross Validation Accuracy vs. Number of Eigenvectors', num_folds));
hold off

outname = '../../results/flda_acc.png';
print(outname, '-dpng')

[ best_acc, best_idx ] = max(mean(train_acc, 2));
mean_test_acc = mean(test_acc, 2);

flda_train = best_acc;
flda_test = mean_test_acc(best_idx);

train_std = std(train_acc, 0, 2);
test_std = std(test_acc, 0, 2);
flda_train_std = train_std(best_idx);
flda_test_std = test_std(best_idx);

fprintf('Best training accuracy: %0.1g%%, by %d-D projection.\n', flda_train, best_idx);
fprintf('The corresponding test accuracy: %0.1g%%.\n', flda_test);

ws_fname = '../../results/flda_ws.mat';
save(ws_fname);
