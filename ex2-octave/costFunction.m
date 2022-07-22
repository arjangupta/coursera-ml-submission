function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

cost_sum = 0;
gradient_sum_vec = zeros(size(theta),1); % a vector of sums

for i = 1:m
    % Calculate the hypothesis
    h = sigmoid(theta'*(X(i,:)'));
    % Add to the cost sum
    cost_sum = cost_sum + ( -y(i)*log(h) - (1-y(i))*log(1-h) );
    % Add to the gradient sum
    gradient_sum_vec = gradient_sum_vec + ( h - y(i) )*X(i,:)';
end

% Set the cost formula
J = (1/m)*cost_sum;

% Set the gradient formula
grad = (1/m)*gradient_sum_vec

% =============================================================

end
