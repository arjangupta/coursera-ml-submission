function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% y is a vector of 1s and 0s.
% Get vector of all indices of 1's in y
positives = find(y);
% Get vector of all indices of 0's in y
invert_y = (y==0);
negatives = find(invert_y);

% Plot Examples
plot(X(positives, 1), X(positives, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(negatives, 1), X(negatives, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

% =========================================================================



hold off;

end