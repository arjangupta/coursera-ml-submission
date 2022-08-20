function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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

C = 0.01;
sigma = 0.01;

% Train and evaluate C and sigma with these initial values
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model, Xval);
err = mean(double(predictions ~= yval));

% Set this as the minimum error seen so far, and best C and sigma so far
min_err = err;
best_C = C;
best_sigma = sigma;

base_multiplicand_C = C;
base_multiplicand_sigma = sigma;

for i = 1:8
    % Change C
    if (rem(i/2) == 0)
        base_multiplicand_C *= 10;
        C = base_multiplicand_C;
    else
        C = 3 * base_multiplicand_C;
    endif

    for j = 1:8
        % Change sigma
        if (rem(j/2) == 0)
            base_multiplicand_sigma *= 10;
            sigma = base_multiplicand_sigma;
        else
            sigma = 3 * base_multiplicand_sigma;
        endif

        % Train and evaluate C and sigma
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));

        % If error is lowest ever, then record this
        if (err < min_err)
            min_err = err;
            best_C = C;
            best_sigma = sigma;
        endif
    end
end

C = best_C;
sigma = best_sigma;

% =========================================================================

end
