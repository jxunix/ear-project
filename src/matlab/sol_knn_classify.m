function [ypred] = sol_knn_classify(Xtrain, ytrain, Xtest, K, dst)
% kNN (for labels in {0, 1}), ties broken using more neighbors
%
% function [ypred] = sol_knn_classify(Xtrain, ytrain, Xtest, K, dst)
%
% input: 
% Xtrain(n,:) = n-th training example (d-dimensional)
% ytrain(n) = n-th training label (integer)
% Xtest(m,:) = m-th test example
% K: number of neighbors
% dst: precomputed pairwise distance matrix [optional]
%
% output:
% ypred(m) = predicted class label for m-th test example

Ntrain = size(Xtrain, 1);
if K >= Ntrain
	fprintf('reducing K from %d to %d\n', K, Ntrain-1);
	K = Ntrain - 1;
end

% note: let caller provide dst matrix, and compute only when not provided
if nargin < 5  
	fprintf('Computing distance matrix: '); tic;
	% Euclidean distance: dst(n,m) = ||Xtrain(n) - Xtest(m)||^2
	dst = pdist2(double(Xtrain), double(Xtest)); 
	% optional: Hamming distance
	% dst = pdist2(Xtrain, Xtest, 'hamming');
	toc;
end

ypred = zeros(size(Xtest,1), 1);
if K == 1
	% find nearest neighbor
	[~, closest] = min(dst, [], 1);
	ypred = ytrain(closest);

else  % K > 1
	for i = 1:size(Xtest, 1)
		% sort training examples according to distance
		[val, idx] = sort(dst(:, i));

		% inspect labels of K nearest neighbors
		% note: it's possible to have ties in distance computation, in that case
		% we'll be taking the K nearest neighbors with smallest indexes
		labels = ytrain(idx(1:K));

		% break ties, if any
		ypred(i) = mode(labels);
	end
	%fprintf('K=%d: tie-breaking performed %d times\n', K, tb_cnt);

end
