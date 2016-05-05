%
% Filename: knn.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Tue 22 Mar 2016 05:09:35 PM EDT
% Description: This script is to classify the feature vectors by applying KNN.
%

add_pixel = 0;

load('../../results/symmetry_test_ws_manual.mat', 'M', 'names');
if add_pixel
	load('../../results/extract_pixel_ws_manual.mat', 'rgbs');
end

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

if add_pixel
	log_eta = -3:3;
	eta = 10 .^ log_eta;
	eta = [ 0 eta ];

	dists_rgb = pdist2(rgbs, rgbs);
	acc_stat = zeros(length(eta), 2);
	
	for i = 1:length(eta)
		fprintf('eta=%0.3g\n', eta(i));
		dists_sum = sqrt(dists .^ 2 + eta(i) * (dists_rgb .^ 2));
		acc = zeros(length(Ks), num_folds);

		for K = 1:length(Ks)
			for fold = 1:num_folds
				train_id = trainfolds{fold};
				test_id = testfolds{fold};

				X_train = X(train_id, :);
				y_train = y(train_id);
				X_test = X(test_id, :);
				y_test = y(test_id);
				ds = dists_sum(train_id, test_id);

				y_pred = sol_knn_classify(X_train, y_train, X_test, K, ds);
				acc(K, fold) = 100 * mean(y_pred == y_test);
			end
		end

		acc_mean = mean(acc, 2);
		acc_std = std(acc, 0, 2);
		[ val idx ] = max(acc_mean);

		acc_stat(i,1) = val;
		acc_stat(i,2) = acc_std(idx);
	end

	bar(acc_stat(:,1), 'FaceColor', [ 0.5, 0.5, 0.5 ]);
	set(gca, 'XtickLabel', { '0', '0.001', '0.01', '0.1', '1', '10', '100', '1000' });
	xlabel('\eta');
	ylabel('Recognition rate %');
	%title('10-fold Cross Validation Accuracy vs \eta');
	hold on
	errorbar(acc_stat(:,1), acc_stat(:,2), '.k');
	hold off

	outname = '../../results/knn_acc_rgb_manual.png';
	print(outname, '-dpng')
    
    ws_fname = '../../results/knn_ws_rgb_manual.mat';
    save(ws_fname);
else
	acc = zeros(length(Ks), num_folds);

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
			acc(K, fold) = 100 * mean(y_pred == y_test);
		end
	end

	% plot cross validation accuracy against K
	errorbar(Ks, mean(acc, 2), std(acc, 0, 2), '-ok');
	xlabel('K')
	ylabel('Recognition rate %')
	title(sprintf('%d-fold Cross Validation Accuracy vs. K', num_folds));

	outname = '../../results/knn_acc_manual.png';
	print(outname, '-dpng')

	[ val, idx ] = max(mean(acc, 2));
	fprintf('Best accuracy is %0.1f%% when K=%d.\n', val, idx)

	knn_test = val;
	stdev = std(acc, 0, 2);
	knn_test_std = stdev(idx);
    
    ws_fname = '../../results/knn_ws_manual.mat';
    save(ws_fname);
end
