function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------
X1=transpose(X); %X matric transpose
X2=X1*X;
X3=inv(X2);
X4=X3*X1;
theta=X4*y;





% -------------------------------------------------------------


% ============================================================

end
