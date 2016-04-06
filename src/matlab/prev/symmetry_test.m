%
% Filename: symmetry_test.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Wed 16 Dec 2015 12:01:09 AM EST
% Description: This script plots the histograms of the matching scores
%		(normalized correlation coefficient and mutual information) between one left
%		ear and all others within the same visit. By comparing the probabilistic
%		distribution of the matching score between two ears of the same person, that
%		of the maximum matching score and that of the average matching score, we
%		test whether the left and right ears are symmetric.
%

for visit_num=1:3
	for ncc=0:1
		if ncc
			fname = [ '../../../results/visit', int2str(visit_num), '/matching_ncc2.csv' ];
			outname = [ '../../../results/visit', int2str(visit_num), '/symmetry_test_ncc.png' ];
		else
			fname = [ '../../../results/visit', int2str(visit_num), '/matching_mi2.csv' ];
			outname = [ '../../../results/visit', int2str(visit_num), '/symmetry_test_mi.png' ];
		end

		M = importdata(fname);
		cols = size(M, 1);

		symm = zeros(cols, 1);
		for i = 1:cols
			symm(i) = M(i, 2*i);
			M(i, 2*i-1) = 0;
			M(i, 2*i) = 0;
		end

		maxi = max(M, [], 2);
		aver = mean(M, 2);

		compare = [ symm maxi aver ];
		[ counts, centers ] = hist(compare);
		area = sum(counts);
		counts_prob = counts ./ repmat(area, 10, 1);
		bar(centers, counts_prob);

		colormap('gray');
		legend('Symmetry', 'Maximum', 'Average', 'Location', 'best');
		if ncc
			title('Histograms of Normalized Correlation Coefficient - Same Visit')
			xlabel('NCC')
		else
			title('Histograms of Mutual Information - Same Visit')
			xlabel('MI')
		end
		ylabel('Probability')

		symm_pd = fitdist(symm(:), 'normal');
		maxi_pd = fitdist(maxi(:), 'normal');
		aver_pd = fitdist(aver(:), 'normal');

		if ncc
			x = 0.0:0.001:1.0;
		else
			x = 0.4:0.001:1.8;
		end

		symm_PDF = pdf(symm_pd, x);
		maxi_PDF = pdf(maxi_pd, x);
		aver_PDF = pdf(aver_pd, x);

		symm_PDF = symm_PDF / area(:,1);
		maxi_PDF = maxi_PDF / area(:,2);
		aver_PDF = aver_PDF / area(:,3);
        
		a=-1;
		b=-1;
		for i=1:size(symm_PDF,2)-1
			if (symm_PDF(:,i) <= maxi_PDF(:,i) & symm_PDF(:,i+1) > maxi_PDF(:,i+1))
				a=x(i);
			end
			if (symm_PDF(:,i) >= maxi_PDF(:,i) & symm_PDF(:,i+1) < maxi_PDF(:,i+1))
				b=x(i);
			end
		end

		hold on
		plot(x, symm_PDF, 'k')
		plot(x, maxi_PDF, 'color', [0.5,0.5,0.5])
		plot(x, aver_PDF, 'k--')
		if a >= 0
			plot([ a a ], ylim, 'r--')
			text(a, 1.5*mean(ylim), [ ' \leftarrow x=', num2str(a) ])
		end
		if b >= 0
			plot([ b b ], ylim, 'r--')
			text(b, 1.5*mean(ylim), [ ' \leftarrow x=', num2str(b) ])
		end
		hold off

		print(outname, '-dpng')
	end
end
