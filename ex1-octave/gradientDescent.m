function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % Compute the current hypothesis
    H = X * theta;

    % Go through and compute the descent for each parameter, store
    % in temporary matrix
    temp_theta = zeros(size(theta))
    for param_i = 1:size(n)

        temp_theta(param_i) = theta(param_i) - (alpha * (1/m) * sum(H-y)*X(:,param_i))

    end

    % Assign the parameters (simultaneous update)
    theta = temp_theta


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
