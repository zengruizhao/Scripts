function [percent,C,ave_planes,stdev] = cross_val(A,B,numgroups,maxdepth,tol,CF,min_points)
% CROSS_VAL determines the percentage of points correctly 
%	classified by the MSM-T algorithm (decision tree), using cross-
%	validation process, the confusion matrix, and the average number
%	of planes needed.
%
%	[percent,C,ave_planes,stdev] = cross_val(A,B,numgroups,maxdepth,tol,CF,min_points)
%
%	percent: percentage of points correctly classified by MSMT
%	C represents the confusion matrix as follows:
%
%				OBSERVED
%
%				A		B
%
%			A  # pts. of A	    # pts. of B	
%	PREDICTED	   properly 	    misclassified
%			   classified
%
%			B  # pts. of A	    # pts. of B
%			   misclassified    properly
%					    classified
%
%	(C is a 2 x 2 matrix)
%
%	aveplane is the average number of planes needed in seperation
%	aveplanes = numplanes/numgroups
%	
%	stdev: standard deviation of percentage correct for each group
%
%	A: m x n matrix representing the m n-dimensional points in set A
%	B: k x n matrix that takes care of the k n-dimensional points in set
%	B.
%	numgroups: number of groups to use for cross-validation
%	maxdepth: maximum depth in decision trees (if not given, entire tree
%		  will be created).
%	tol: given tolerance such that a given region will not be 
%	separated if either: 1. it contains only points of A or only points of
%			        B.
%			     2. the percentage error in that region is less 
%				than tol.
%
%	CF: certainty factor used in the C4.5 pruning algorithm (Quinlan)
%		if not given or CF = 0.0, the tree will not be pruned using
%		the C4.5 pruning algorithm
%	min_points: minimum number of points such that a node is pruned
%		if the number of misclassified points at that node is less
%		than min_points
%		if not given or min_points = 0, the tree will not be pruned
%
%	outline of cross-validation process:
%		1. shuffle the rows of A,B.
%		2. determine groups to be used in crossvalidation
%		3. for all the groups:
%			i. determine the correct test, train matrices
%			ii. train (build tree)
%			iii. test (obtain total number of errors)
%		4. determine percentage of correctly classified points.
%		   determine confusion matrix C.
%		   determine average number of planes needed.
global n;

if nargin < 7
	min_points = 0;
end
if nargin < 6
	CF = 0.0;
end
if nargin < 5
	tol = 0.0;
end
if nargin < 4 
	maxdepth = round(realmax);
end
n = size(A,2);
% shuffle the decks:
A = shuffle(A);
B = shuffle(B);
%
% think up the groups:
Agroups = cross_makegroups(size(A,1),numgroups);
Bgroups = cross_makegroups(size(B,1),numgroups);
%
% do it:
stanvec = [];
totmissA = 0;
totmissB = 0;
totcorrectA = 0;
totcorrectB = 0;
numplanes = 0;
% print out a header
disp(['Count' blanks(2) 'Total Correct' blanks(2) 'Total Incorrect' blanks(2) 'Percentage Correct'])
disp(blanks(1)')

for i = 1:numgroups

	[Atest,Btest,Atrain,Btrain] = cross_sets(A,B,Agroups,Bgroups,i);

% Train! (grow the Tree for these training points)

	T = [];
	T = msmt_tree(Atrain,Btrain,maxdepth,tol,CF,min_points);
	numplanes = numplanes + size(T,2); % number of planes is the 
					      % number of columns in Tree

% Test! (test with the test points)

	[missA,correctA,missB,correctB] = cross_classify(T,Atest,Btest);
	totmissA = totmissA + missA;
	totmissB = totmissB + missB;
	totcorrectA = totcorrectA + correctA;
	totcorrectB = totcorrectB + correctB;
	pc = 100*((correctA + correctB)/(missA + missB + correctA + correctB));
	stanvec(i) = pc;

% Print out some information regarding this test:
	disp([ num2str(i) blanks(7) sprintf('%7.3f ',correctA + correctB) blanks(6) sprintf('%7.3f ',missA + missB) blanks(9) sprintf('%7.3f ',pc)])
		
	Atest = A(Agroups(i):Agroups(i+1)-1,:);
	Btest = B(Bgroups(i):Bgroups(i+1)-1,:);
	Atrain = [ A(1:Agroups(i)-1,:);A(Agroups(i+1):size(A,1),:)];
	Btrain = [ B(1:Bgroups(i)-1,:);B(Bgroups(i+1):size(B,1),:)];
end % for

% compute percentage correctly classified:

percent = ((totcorrectA + totcorrectB)/(size(A,1)+size(B,1)))*100;
C = [ totcorrectA, totmissB;
      totmissA, totcorrectB ];
ave_planes = numplanes/numgroups;
stdev = std(stanvec);







