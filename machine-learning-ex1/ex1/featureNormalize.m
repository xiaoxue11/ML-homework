function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%  
mu=mean(X); %the mean of every column.
sigma=std(X); %the std of every colum.
mu1=mu(1,1);
mu2=mu(1,2);
sigma1=sigma(1,1);
sigma2=sigma(1,2);
X1=X(:,1);
X2=X(:,2);
norm1=(X1-mu1)/sigma1;
norm2=(X2-mu2)/sigma2;
X_norm=[norm1,norm2];
% ============================================================

end
