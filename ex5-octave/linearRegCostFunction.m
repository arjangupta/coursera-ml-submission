function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

size_X = size(X)
size_y = size(y)
size_theta = size(theta)
size_lambda = size(lambda)

reg_theta = [zeros(1,size(theta,2)); theta(2:end,:)];

% Regularized cost
H = X*theta;
J = (1/(2*m))*sum((H-y).^2) + (lambda/(2*m))*sum(reg_theta.^2);

% Regularized gradient
grad = (1/m)*sum((H-y)'*X) + (lambda/(2*m))*reg_theta;


% =========================================================================

grad = grad(:);

end
