%
% Filename: svm_binary.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Tue 29 Mar 2016 01:28:42 PM EDT
% Description: This script is to implement binary SVM in Matlab.
%

function [ y_pred_tt, y_prob_tt, y_pred_tr, y_prob_tr ] = svm_binary(X_train, y_train, X_test, label_train, rindex_train, rindex_test, penalize, softmargin, lambda)

%function y_pred_tt = svm_binary()
%load xdata2d

%softmargin = 1;
%X_train = Xtrain;
%X_test = Xtest;
%y_train = assign_labels(X_train, 1);
%y_test = assign_labels(X_test, 1);

X_train = X_train';
X_test = X_test';
y_train = y_train';
y_train = 2 * y_train - 1;

%%
[ dim, N ] = size(X_train);
options = optimset('GradObj', 'on', 'MaxFunEvals', 1e6, 'MaxIter', 100, 'Display', 'off');

% we need access to these variables during the optimization
assignin('base', 'y_selected', y_train);
assignin('base', 'X_selected', X_train);
assignin('base', 'label_selected', label_train);
assignin('base', 'rindex_train_selected', rindex_train);
assignin('base', 'rindex_test_selected', rindex_test);
assignin('base', 'lambda', lambda);
assignin('base', 'N', N);
assignin('base', 'dim', dim);

if softmargin == 0
	% create matrix A, vector b for linear inequalities
	A = X_train .* repmat(y_train, dim, 1);
	A = [ A; y_train ];
	A = A';
	b = ones(N,1);
	x0 = zeros(1, dim+1);

	% the fmincon function requires: c <= 0!
	A = -A;
	b = -b;

	% start
	if penalize == 1
		[ vec, fval, exitflag ] = fmincon(@normal_and_penalty, x0, A,b,[],[],[],[],[], options);
	else
		[ vec, fval, exitflag ] = fmincon(@normal, x0, A,b,[],[],[],[],[], options);
	end
else
	% in case of the softmargin, we have nonlinear inequalities and need to
	% provide a dedicated function to evaluate the constraint at each iteration
	x0 = zeros(1, N+dim+1);

	[ vec, fval, exitflag ] = fmincon(@normal_and_slack, x0, [],[],[],[],[],[], @slack_con, options);
end

% classify test patterns
y_prob_tr = vec(1:dim) * X_train + vec(dim+1);
y_prob_tr = y_prob_tr';
y_pred_tr = y_prob_tr > 0;

y_prob_tt = vec(1:dim) * X_test + vec(dim+1);
y_prob_tt = y_prob_tt';
y_pred_tt = y_prob_tt > 0;

%assignin('base', 'y_prob_tr', y_prob_tr);
%assignin('base', 'y_pred_tr', y_pred_tr);
%assignin('base', 'y_prob_tt', y_prob_tt);
%assignin('base', 'y_pred_tt', y_pred_tt);

end

% objective function for hard margins
function [ loss, g ] = normal(x0)
	omega = x0(1:end-1);
	loss = 1/2 * norm(omega)^2;
	g = [ omega, 0 ];
end

% objective function with penalty for hard margins
function [ loss, g ] = normal_and_penalty(x0)
	lambda = evalin('base', 'lambda');
	X_train = evalin('base', 'X_selected');
	label_train = evalin('base', 'label_selected');
	rindex_train = evalin('base', 'rindex_train');
	rindex_test = evalin('base', 'rindex_test');
	N = evalin('base', 'N');

	omega = x0(1:end-1);
	b = x0(end);
	y_prob_tr = omega * X_train + b;
	y_pred_tr = y_prob_tr > 0;
	%y_pred_tr2 = y_pred_tr;
	y_pred_tr = double(y_pred_tr);
	y_pred_tr(rindex_train) = y_pred_tr;
	y_pred_tr(rindex_test) = -1;
	penalty = 0;

	% penalize the prediction inconsistency
	for loop = 1:N/4
		l1 = y_pred_tr(4*loop-3);
		l2 = y_pred_tr(4*loop-2);
		r1 = y_pred_tr(4*loop-1);
		r2 = y_pred_tr(4*loop);

		if l1 ~= -1 & l2 ~= -1
			penalty = penalty + (l1 ~= l2);
		end
		if l2 ~= -1 & r1 ~= -1
			penalty = penalty + (l2 ~= r1);
		end
		if r1 ~= -1 & r2 ~= -1
			penalty = penalty + (r1 ~= r2);
		end
		if l1 ~= -1 & r2 ~= -1
			penalty = penalty + (l1 ~= r2);
		end
	end

	loss = 1/2 * norm(omega)^2 + lambda * penalty;
	g = [ omega, 0 ];
end

% objective function for soft margins with slack variables
function [ loss, g ] = normal_and_slack(x0)
	C = 1;
	N = evalin('base', 'N');
	dim = evalin('base', 'dim');

	omega = x0(1:dim);
	xi = x0(dim+2:end);
	loss = 1/2 * norm(omega)^2 + C * sum(xi);
	g = [ omega, 0, C * ones(1,N) ];
end

% nonlinear constraints for soft margin problem with slack variables
function [ c, ceq ] = slack_con(x0)
	y_train = evalin('base', 'y_selected');
	X_train = evalin('base', 'X_selected');
	N = evalin('base', 'N');
	dim = evalin('base', 'dim');

	omega = x0(1:dim);
	b = x0(dim+1);
	xi = x0(dim+2:end);

	c = omega * X_train + b;
	c = y_train .* c;
	c = c - 1 + xi;
	c = [ -c -xi ]';
	ceq = [];
end
