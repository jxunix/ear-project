%
% Filename: retrieval_analysis.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Tue 15 Dec 2015 11:52:16 PM EST
% Description: This script is to compute the precision/recall curve and ROC
%		curve.
%

ncc = 0;
visit_num = 3;

if ncc
	fname = [ '../../../results/visit', int2str(visit_num), '/matching_ncc2.csv' ];
	outname = [ '../../../results/visit', int2str(visit_num), '/retrieval_analysis_ncc.png' ];
else
	fname = [ '../../../results/visit', int2str(visit_num), '/matching_mi2.csv' ];
	outname = [ '../../../results/visit', int2str(visit_num), '/retrieval_analysis_mi.png' ];
end

M = importdata(fname);
cols = size(M, 1);

symm = zeros(cols, 1);
for i = 1:cols
	symm(i) = M(i, 2*i);
	M(i, 2*i-1) = 0;
end

top_n = sort(M, 2, 'descend');

n = 2*cols - 1;
is_in = zeros(cols, n);
for i=1:n
	is_in(:,i) = (top_n(:,i) <= symm);
end

tp = mean(is_in);
test_pos = linspace(1,n,n);

recall = tp / 1;
precision = tp ./ test_pos;
fpr = (test_pos - tp) / (n-1);

recall = [ 0 recall ];
precision = [ 1 precision ];
fpr = [ 0 fpr ];

figure
subplot(1,2,1);
plot(recall, precision, '-k');
if ncc
	title('Precision Recall Curve - NCC')
else
	title('Precision Recall Curve - MI')
end
xlabel('Recall (TPR)')
ylabel('Precision')

subplot(1,2,2);
plot(fpr, recall, '-k');
if ncc
	title('ROC Curve - NCC')
else
	title('ROC Curve - MI')
end
xlabel('FPR')
ylabel('TPR')

print(outname, '-dpng')
