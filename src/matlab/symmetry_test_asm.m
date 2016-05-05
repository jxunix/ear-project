%
% Filename: symmetry_test_asm.m
% Author: Jun Xu
% Email: junx@cs.bu.edu
% Created Time: Wed 16 Mar 2016 02:53:04 PM EDT
% Description: This script is to perform the test for ear symmetry.
%

%% read the image filenames
fname = '../../results/image_names.txt';
names = textread(fname, '%s', 'delimiter', '\n');

%% read the coordinates of the feature points and remove missing data (whose
%  corresponding filename contains '(remedy)'.
%fname = '../../results/features_asm.csv';
fname = '../../results/features_manual.csv';
M = csvread(fname);

rows = size(M,1);
index = zeros(rows, 1);
for i = 1:rows
	s = size(strfind(names{ceil(i/2)}, 'remedy'), 2);
	
	if s == 1
		i = ceil(i/8)*8+1;
		index(i-8) = 1;
		index(i-7) = 1;
		index(i-6) = 1;
		index(i-5) = 1;
		index(i-4) = 1;
		index(i-3) = 1;
		index(i-2) = 1;
		index(i-1) = 1;
	end
end

index = logical(index);
M = M(~index, :);

index = index(1:2:length(index));
names = names(~index);

%% standardize all feature vectors without scaling
M = align(M);
rows = size(M,1) / 2;

%% test for ear symmetry for the same newborn at the same visit
X = reshape(M', size(M, 2)*2, rows)';
dists = pdist2(X, X);

% create a mask to pick those matching scores at the same visit
mask = zeros(rows, rows);
for i = 1:rows
	for j = 1:rows
		if names{i}(10) == names{j}(10)
			mask(i,j) = 1;
		end
	end
end

dists_same = dists .* mask;
self = zeros(rows, 1);
symm = zeros(rows, 1);
for i = 1:rows
	if mod(i,2) == 1
		self(i,1) = dists_same(i, i+1);
		dists_same(i, i+1) = 0;
	else
		self(i,1) = dists_same(i, i-1);
		dists_same(i, i-1) = 0;
	end
	if mod(i,4) == 1
		symm(i,1) = max([dists_same(i, i+2), dists_same(i, i+3)]);
	elseif mod(i,4) == 2
		symm(i,1) = max([dists_same(i, i+1), dists_same(i, i+2)]);
	elseif mod(i,4) == 3
		symm(i,1) = max([dists_same(i, i-2), dists_same(i, i-1)]);
	else
		symm(i,1) = max([dists_same(i, i-3), dists_same(i, i-2)]);
	end
end
maxi = max(dists_same, [], 2);

%% plot histogram of the matching score between ears for the same newborn at the same visit
data = [ self symm maxi ];
[ counts, centers ] = hist(data);
bar(centers, counts);

colormap('gray');
legend('Left-Left', 'Left-Right', 'Left-Others''', 'Location', 'best');
%title('Histogram of the Matching Scores between the Ear Images Taken at The Same Visit')
xlabel('Euclidean Distance')
ylabel('Counts')

self_pd = fitdist(self, 'normal');
symm_pd = fitdist(symm, 'normal');
maxi_pd = fitdist(maxi, 'normal');

x = 0:0.5:ceil(centers(end) / 500 + 1) * 500;
self_pdf = pdf(self_pd, x);
symm_pdf = pdf(symm_pd, x);
maxi_pdf = pdf(maxi_pd, x);

area = rows * (centers(2) - centers(1));
self_pdf = self_pdf * area;
symm_pdf = symm_pdf * area;
maxi_pdf = maxi_pdf * area;

a = -1;
for i = 1:size(symm_pdf, 2)-1
	if (symm_pdf(:,i) >= maxi_pdf(:,i) & symm_pdf(:,i+1) < maxi_pdf(:,i+1))
		a = x(i);
	end
end

hold on
plot(x, self_pdf, 'k')
plot(x, symm_pdf, 'color', [ 0.5,0.5,0.5 ])
plot(x, maxi_pdf, 'k--')

if a >= 0
	plot([ a a ], ylim, 'r--')
	text(a, 1.5*mean(ylim), [ '\leftarrow x=', num2str(a) ])
end
hold off

<<<<<<< HEAD
outname = '../../results/symmetry_test_same_visit_manual.png';
print(outname, '-dpng')
=======
%outname = '../../results/symmetry_test_same_visit_asm.png';
%print(outname, '-dpng')
>>>>>>> f4f9f80a0610fe7036476af73f13f568389cda2b

%% clean the data for test for ear symmetry for the same newborn
%  between two consecutive visits.
mask = zeros(rows, rows);
for i = 1:rows
	for j = 1:rows
		if names{i}(10) + 1 == names{j}(10)
			mask(i,j) = 1;
		end
	end
end

dists_next = dists .* mask;
self = zeros(rows, 1);
symm = zeros(rows, 1);
for i = 1:rows
	for j = 1:rows
		if (names{i}(1:3) == names{j}(1:3) & names{i}(10) + 1 == names{j}(10) & names{i}(12) == names{j}(12))
			self(i) = max([ self(i), dists_next(i,j) ]);
		end
		if (names{i}(1:3) == names{j}(1:3) & names{i}(10) + 1 == names{j}(10) & names{i}(12) ~= names{j}(12))
			symm(i) = max([ symm(i), dists_next(i,j) ]);
		end
	end
end
maxi = max(dists_next, [], 2);

maxi = maxi(self ~= 0);
self = self(self ~= 0);
symm = symm(symm ~= 0);

%% plot histogram of matching score between ears for the same newborn between
%  two consecutive visits.
data = [ self symm maxi ];
[ counts, centers ] = hist(data);
bar(centers, counts);

colormap('gray');
legend('Left-Left', 'Left-Right', 'Left-Others''', 'Location', 'best');
%title('Histogram of the Matching Scores between the Ear Images Taken at Two Consecutive Visits')
xlabel('Euclidean Distance')
ylabel('Counts')

self_pd = fitdist(self, 'normal');
symm_pd = fitdist(symm, 'normal');
maxi_pd = fitdist(maxi, 'normal');

x = 0:0.5:ceil(centers(end) / 500 + 1) * 500;
self_pdf = pdf(self_pd, x);
symm_pdf = pdf(symm_pd, x);
maxi_pdf = pdf(maxi_pd, x);

area = length(maxi) * (centers(2) - centers(1));
self_pdf = self_pdf * area;
symm_pdf = symm_pdf * area;
maxi_pdf = maxi_pdf * area;

a = -1;
for i = 1:size(symm_pdf, 2)-1
	if (symm_pdf(:,i) >= maxi_pdf(:,i) & symm_pdf(:,i+1) < maxi_pdf(:,i+1))
		a = x(i);
	end
end

hold on
plot(x, self_pdf, 'k')
plot(x, symm_pdf, 'color', [ 0.5,0.5,0.5 ])
plot(x, maxi_pdf, 'k--')

if a >= 0
	plot([ a a ], ylim, 'r--')
	text(a, 1.5*mean(ylim), [ '\leftarrow x=', num2str(a) ])
end
hold off

<<<<<<< HEAD
outname = '../../results/symmetry_test_next_visit_manual.png';
print(outname, '-dpng')

ws_fname = '../../results/symmetry_test_ws_manual.mat';
save(ws_fname);
=======
%outname = '../../results/symmetry_test_next_visit_asm.png';
%print(outname, '-dpng')

%ws_fname = '../../results/symmetry_test_ws_asm.mat';
%save(ws_fname);
>>>>>>> f4f9f80a0610fe7036476af73f13f568389cda2b
