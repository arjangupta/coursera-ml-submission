function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% Compute cost and gradient sum terms
cost_sum = 0;
grad_sum_vec = zeros(size(theta),1);
for i = 1:m
    Xi = X(i,:)';
    h_Xi = sigmoid(theta'*Xi);
    cost_sum = cost_sum + ( -y(i) * log( h_Xi ) - (1 - y(i)) * log( 1 - h_Xi ) ); % formula for cost sum
    grad_sum_vec = grad_sum_vec + ( h_Xi - y(i) ) * Xi; % should give a vector of 28
end

% Compute regularization sum term for cost
theta_sq_sum = 0;
n = size(theta, 1);
for i = 1:n
    theta_sq_sum = theta_sq_sum + ( theta(i)^2 );
end

% Compute the cost
J = (1/m) * cost_sum + (lambda/(2*m)) * theta_sq_sum;

% Compute the gradient without regularization
grad = (1/m)*grad_sum_vec;

% Add the regularization term
for i = 2:size(grad)
    grad(i) = grad(i) + (lambda/m)*theta(i);
end

% =============================================================

end
