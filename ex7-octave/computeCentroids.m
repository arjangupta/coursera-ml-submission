function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% idx (m x 1) is the vector that stores the centroid index (can take on values 1:K) of each example X(i)

num_examples_per_centroid = zeros(K,1);

% Calculate the sums
for i = 1:m
    for j = 1:K
        % Go through all examples and add to a running sum
        if (idx(i) == j)
            centroids(j,:) = centroids(j,:) + X(i,:);
            num_examples_per_centroid(j) += 1;
        endif
    end
end

% Calculate the means
for j = 1:K
    if (num_examples_per_centroid != 0)
        centroids(j,:) = centroids(j,:) / num_examples_per_centroid(j);
    else
        centroids(j,:) = 0;
    endif
end

% =============================================================


end

