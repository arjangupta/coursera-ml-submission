function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Bias units
column_of_ones = [ones(m, 1)];

% Feedforward propagation
layer1_result = sigmoid([column_of_ones, X]*Theta1'); % should give 5000 x 25 matrix
layer2_result = sigmoid([column_of_ones, layer1_result]*Theta2'); % should give 5000 x 10 matrix

% Find the result of each training example
[val, idx] = max(layer2_result, [], 2);

% Return the predictions
p = idx;

% =========================================================================


end
