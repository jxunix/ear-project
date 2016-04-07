%
% Filename: summary.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Wed 06 Apr 2016 04:39:25 PM EDT
% Description: This script summarize all the recognition rates.
%

clear; close all; symmetry_test_asm
clear; close all; knn
clear; close all; flda
clear; close all; svm
clear; close all; svm_interface

clear; close all;

fname = '../../results/knn_ws.mat';
variables = { 'acc' };
ws = load(fname, variables{:});
knn_test_acc = ws.('acc');

fname = '../../results/flda_ws.mat';
variables = { 'train_acc', 'test_acc' };
ws = load(fname, variables{:});
flda_train_acc = ws.('train_acc');
flda_test_acc = ws.('test_acc');

fname = '../../results/svm_ws.mat';
variables = { 'train_acc', 'test_acc' };
ws = load(fname, variables{:});
svm_train_acc = ws.('train_acc');
svm_test_acc = ws.('test_acc');

fname = '../../results/svm_own_ws.mat';
variables = { 'train_acc', 'test_acc' };
ws = load(fname, variables{:});
svm_own_train_acc = ws.('train_acc');
svm_own_test_acc = ws.('test_acc');

clear fname variables ws;

knn_test_acc_mean = mean(knn_test_acc, 2);
knn_test_acc_std = std(knn_test_acc, 0, 2);

flda_train_acc_mean = mean(flda_train_acc, 2);
flda_test_acc_mean = mean(flda_test_acc, 2);
flda_train_acc_std = std(flda_train_acc, 0, 2);
flda_test_acc_std = std(flda_test_acc, 0, 2);

svm_train_acc_mean = mean(svm_train_acc, 2);
svm_test_acc_mean = mean(svm_test_acc, 2);
svm_train_acc_std = std(svm_train_acc, 0, 2);
svm_test_acc_std = std(svm_test_acc, 0, 2);

svm_own_train_acc_mean = mean(svm_own_train_acc, 2);
svm_own_test_acc_mean = mean(svm_own_test_acc, 2);
svm_own_train_acc_std = std(svm_own_train_acc, 0, 2);
svm_own_test_acc_std = std(svm_own_test_acc, 0, 2);

[ val idx ] = max(knn_test_acc_mean);
knn_train = 100;
knn_train_std = 0;
knn_test = val;
knn_test_std = knn_test_acc_std(idx);

[ val idx ] = max(flda_train_acc_mean);
flda_train = val;
flda_test = flda_test_acc_mean(idx);
flda_train_std = flda_train_acc_std(idx);
flda_test_std = flda_test_acc_std(idx);

[ val idx ] = max(svm_train_acc_mean);
mean_max_arr = svm_test_acc_mean(svm_train_acc_mean == val);
std_max_arr = svm_test_acc_std(svm_train_acc_mean == val);
[ val2 idx2 ] = max(mean_max_arr);
svm_train = val;
svm_test = val2;
svm_train_std = svm_train_acc_std(idx);
svm_test_std = std_max_arr(idx2);

[ val idx ] = max(svm_own_train_acc_mean);
svm_own_train = val;
svm_own_test = svm_own_test_acc_mean(idx);
svm_own_train_std = svm_own_train_acc_std(idx);
svm_own_test_std = svm_own_test_acc_std(idx);

train = [ knn_train flda_train svm_train svm_own_train ];
test = [ knn_test flda_test svm_test svm_own_test ];
train_std = [ knn_train_std flda_train_std svm_train_std svm_own_train_std ];
test_std = [ knn_test_std flda_test_std svm_test_std svm_own_test_std ];

%train = [ knn_train flda_train svm_train ];
%test = [ knn_test flda_test svm_test ];
%train_std = [ knn_train_std flda_train_std svm_train_std ];
%test_std = [ knn_test_std flda_test_std svm_test_std ];

means = [ train; test ];
stds = [ train_std; test_std ];

errorbar_groups(means', stds', ...
	'bar_width', 0.75, ...
	'errorbar_width', 0.5, ...
	'bar_names', {'Training', 'Test'});
ylabel('Recognition Rate %');
title('Comparison of Feature Classification Methods');
legend('KNN', 'FLDA', 'ASM (libsvm)', 'ASM (own)');

outname = '../../results/classification_comparision.png';
print(outname, '-dpng');

save('../../results/summary_ws.mat');
