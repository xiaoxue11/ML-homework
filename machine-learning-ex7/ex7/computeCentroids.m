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
n1=1;
n2=1;
n3=1;
for i=1:m
    if(idx(i)==1)
        X1(n1,:)=X(i,:);
        n1=n1+1;
    end
    if(idx(i)==2)
        X2(n2,:)=X(i,:);
        n2=n2+1;
    end
    if(idx(i)==3)
        X3(n3,:)=X(i,:);
        n3=n3+1;
    end 
end
u1=mean(X1);
u2=mean(X2);
u3=mean(X3);
centroids=[u1;u2;u3];
% =============================================================
end

