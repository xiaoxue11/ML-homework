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
%=====cost function ==================
    predict1=X*theta;
    g1 = sigmoid(predict1);
    g2=1-g1;
    cost1=log(g1);
    cost2=log(g2);
    cost=y.*cost1+cost2-y.*cost2;
    J=(-1)/m*sum(cost);
    deviation=g1-y;
    Y=transpose(X);
    grad=1/m*(Y*deviation);
% =============================================================

end
