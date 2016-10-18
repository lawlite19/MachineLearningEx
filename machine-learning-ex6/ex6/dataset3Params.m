function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

CTest = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmaTest = [0.01 0.03 0.1 0.3 1,3 10 30];
errors = zeros(length(CTest),length(sigmaTest));

for i = 1:length(CTest)
    for j = 1:length(sigmaTest)
        model= svmTrain(X, y, CTest(i), @(x1, x2) gaussianKernel(x1, x2, sigmaTest(j)));
        predictions = svmPredict(model, Xval);
        errors(i,j) = mean(double(predictions ~= yval));
        clear model;
        clear predictions;
    end
end

[dummy,ind] = min(errors(:));
[i,j] = ind2sub([size(errors,1) size(errors,2)],ind);
C = CTest(i);
sigma = sigmaTest(j);



% =========================================================================

end
