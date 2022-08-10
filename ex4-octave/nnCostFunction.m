function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% ------------ Part 1 ----------------

% 'Widen' the y, as in make 2 become [0 1 0 0 ... 0], etc
Y_wide = eye(num_labels)(y,:); % basically, rearrange an identity matrix of 

% Compute H
X = [ones(m, 1), X]; % pre-pend a column of 1s (bias units)
hidden_layer = sigmoid( X * Theta1'); % X is m x num_features, Theta is num_hidden_units x (num_features + 1)
hidden_layer = [ones(m, 1), hidden_layer]; % prepend bias units as a row
H = sigmoid(hidden_layer * Theta2');

% Iteratively compute cost
for i = 1:m
for j = 1:num_labels
J = J + ( Y_wide(i,j) * log(H(i,j)) ) + ( 1 - Y_wide(i,j) ) * log( 1 - H(i,j) );
end
end
% Multiply the remaining factor for the unregularized cost
J = (-1/m) * J;
% Calculate the regularization
% Sum up all Theta1 elements
sum_theta1_sqs = 0;
for j = 1:size(Theta1,1)
for k = 2:(size(Theta1,2)) %omit the first, bias column
    sum_theta1_sqs = sum_theta1_sqs + Theta1(j,k).^2;
end
end
sum_theta2_sqs = 0;
for j = 1:size(Theta2,1)
for k = 2:(size(Theta2,2)) % omit the first, bias column
    sum_theta2_sqs = sum_theta2_sqs + Theta2(j,k).^2;
end
end
sq_thetas_sum = (sum_theta1_sqs + sum_theta2_sqs);
reg_term = (lambda/(2*m))*(sum_theta1_sqs + sum_theta2_sqs);
J = J + reg_term;

% ------------ Part 2 - Backpropagation ----------------

% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
% Y_wide has size 5000 x 10

big_delta_2 = 0;
big_delta_1 = 0;
for t = 1:m
    % Step 1 - feedforward pass with current training example x(t)
    a1 = X(t,:); % row t of X with pre-pended bias unit - dim 1 x 401
    z2 = a1 * Theta1'; % dim 1 x 25
    z2 = [1 z2]; % place bias unit - dim 1 x 26
    a2 = sigmoid( z2 ); % dim 1 x 26
    z3 = a2 * Theta2';
    a3 = sigmoid( z3 ); % a3 should have dim 1x10
    % Step 2 - output layer diff
    delta_3 = a3 - Y_wide(t); % dim 1 x 10
    % Step 3 - find hidden layer delta
    delta_2 = (delta_3*Theta2).*sigmoidGradient(z2); % dim 1 x 26
    % Step 4 - accumulate the big_deltas
    % For first layer
    delta_2 = delta_2(2:end); % remove bias unit - dim is now 1 x 25
    big_delta_1 = big_delta_1 + ( delta_2' * a1 ); % needs to be same dim as Theta1 - 25 x 401
    % For second layer
    big_delta_2 = big_delta_2 + ( delta_3' * a2 ); % needs to be same dim as Theta2 - 10 x 26
end

Theta1_grad = (1/m)*big_delta_1;
Theta2_grad = (1/m)*big_delta_2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
