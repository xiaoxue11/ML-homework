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
%=====cost function ==================
    predict1=X*theta;
    g1 = sigmoid(predict1);
    g2=1-g1;
    cost1=log(g1);
    cost2=log(g2);
    cost=y.*cost1+cost2-y.*cost2;
    J1=(-1)/m*sum(cost);
    theta1=sum(theta.^2)-theta(1)^2;
    J=J1+lambda/(2*m)*theta1;
    
    deviation=g1-y;
    Y=transpose(X);
    grad1=1/m*(Y*deviation);
    m1=size(theta);
    grad(1)=grad1(1);
    for i=2:m1
        grad(i)=grad1(i)+lambda/m*theta(i);
    end
% =============================================================
end
