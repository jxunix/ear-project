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

load('work_space.mat', 'M', 'names');

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

fold = 1;
train_id = trainfolds{fold};
test_id = testfolds{fold};

X_train = X(train_id, :);
y_train = y(train_id);
X_test = X(test_id, :);
y_test = y(test_id);

[ log2c, log2g ] = meshgrid(-10:10, -10:10);
cv_acc = zeros(numel(log2c), 1);

for i = 1:numel(log2c)
	% train SVM with RBF kernel
	% -s 0: learn classification SVM
	% -t 2: use Gaussian kernel
	% -g  : specify bandwidth
	% -v  : do cross-validation
	% -q  : suppress console output
	libsvm_options = sprintf('-s 0 -t 2 -c %g -g %g -v %d -q', ...
		2^log2c(i), 2^log2g(i), 5);
	cv_acc(i) = svmtrain(y_train, X_train, libsvm_options);
end

[ val, idx ] = max(cv_acc);
fprintf('Best training accuracy: %0.1g%% when C=2^%d, gamma=2^%d.\n', val, log2c(idx), log2g(idx));

%% plot
contour(log2c, log2g, reshape(cv_acc, size(log2c)));
colorbar, hold on
plot(log2c(idx), log2g(idx), 'rx')
text(log2c(idx), log2g(idx), sprintf('Accuracy=%.1f%%', cv_acc(idx)), ...
  'HorizontalAlign', 'left', 'VerticalAlign', 'top');
xlabel('log_2(C)'), ylabel('log_2(\gamma)')
title('10-fold Cross Validation Accuracy vs log_2(C) and log_2(\gamma)')

outname = '../../results/svm_acc.png';
print(outname, '-dpng')

%% train RBF SVM w/ best parameters
libsvm_options = sprintf('-s 0 -t 2 -c %g -g %g -q', 2^log2c(idx), 2^log2g(idx));
model = svmtrain(y_train, X_train, libsvm_options);

% test
y_pred = svmpredict(y_test, X_test, model);
test_acc = 100 * mean(y_pred == y_test);

fprintf('The corresponding test accuracy: %0.1g%%.\n', test_acc);

rr_fname = './recognition_rate.mat';
svm_train = val;
svm_test = test_acc;
save(rr_fname, 'svm_train', '-append');
save(rr_fname, 'svm_test', '-append');
