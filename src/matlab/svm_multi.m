%
% Filename: svm_multi.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Tue 29 Mar 2016 09:36:10 PM EDT
% Description: This script is to implement multi-class SVM.
%

function [ y_pred, y_pred_debug, y_prob, train_err ] = svm_multi(X_train, y_train, X_test, label_train, classes, rindex_train, rindex_test, penalize, lambda)

if nargin < 9
	lambda = -1;
end

y_preds_tr = zeros(size(X_train, 1), classes);
y_probs_tr = zeros(size(X_train, 1), classes);
y_preds_tt = zeros(size(X_test, 1), classes);
y_probs_tt = zeros(size(X_test, 1), classes);

for k = 1:classes
%k = 1;
	fprintf('classify class %d and the rest\n', k)
	y_ova = y_train;
	y_ova(find(y_ova ~= k)) = 0;
	y_ova(find(y_ova == k)) = 1;

	%%
	[ y_pred_tt, y_prob_tt, y_pred_tr, y_prob_tr ] = svm_binary(X_train, y_ova, X_test, label_train, rindex_train, rindex_test, penalize, 0, lambda);
	y_preds_tr(:,k) = y_pred_tr;
	y_probs_tr(:,k) = y_prob_tr;
	y_preds_tt(:,k) = y_pred_tt;
	y_probs_tt(:,k) = y_prob_tt;
end

[ ~, y_pred_tr ] = max(y_probs_tr, [], 2);
train_err = mean(y_pred_tr ~= y_train);

[ ~, y_pred_tt ] = max(y_probs_tt, [], 2);
y_pred = y_pred_tt;

y_pred_debug = y_preds_tt;
y_prob = y_probs_tt;
