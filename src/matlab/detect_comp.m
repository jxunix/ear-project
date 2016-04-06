%
% Filename: detect_comp.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Mon 04 Apr 2016 02:57:14 PM EDT
% Description: This script is to compare the two detection methods, namely,
%   Active Shape Model (ASM) and Generalized Template Matching (GTM).
%   The metric for comparsion is intersection over union (IOU).
%

fname = '../../results/detection_comparision_iou.csv';
M = csvread(fname, 1, 1);

bar(mean(M), 'FaceColor', [ 0.5,0.5,0.5 ]);
set(gca, 'XTickLabel', {'ASM', 'GTM'})
ylabel('IOU')
title('Comparison of Ear Detection Methods')
hold on
errorbar(mean(M), std(M), '.k');
hold off

outname = '../../results/detection_comparision_iou.png';
print(outname, '-dpng')
