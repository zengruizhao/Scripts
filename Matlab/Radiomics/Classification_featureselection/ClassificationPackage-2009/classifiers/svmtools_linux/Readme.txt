First d/l all the files under svmtools.  It has all the files you need to run the svm. The main program you need to run from is called run_demo.m


You input your data just like with the dimensionality reduction (observations x features) except that you also need to input labels eg: [-1 -1 -1 1 1 1]'

currently, it'll simply output an accuracy and its 'strength' of prediction (distance away from the decision hyperplane). 
eg: if the output is -4.3, this means the svm has decided it is very likely to be in the negative class.

You are also allowed to change the amount of your data that is decidated to your training class... default takes the first 1/3 of each class as training.

% commit a portion of the dataset for training
a_values = 1:(size(a,1)/3)   %commit a portion of positive samples for training
b_values = 1:(size(b,1)/3)   %commit a portion of negative samples for training

...as well as change the classification kernels under the filename cv_svm.m

cmd = [ 'svmtrain -t 1 -c ' num2str(2^c) ' -g ' num2str(2^g) ' -v 3 ' train_pathname '_cv.scale' ];

it is currently set to classify under a simple linear kernel noted by -t 1 -c.  
you can also change it to classify using a radial basis function -t 2 -c, or a polynomial -t 3 -c.

For more information, go to http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf

- If that doesn't help... geolee@eden.rutgers.edu