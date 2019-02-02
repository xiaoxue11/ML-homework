function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %
    pred=X*theta;
    deviation=pred-y;
    X1=X(:,1);
    X2=X(:,2);
    X3=X(:,3);
    Y1=deviation.*X1;
    Y2=deviation.*X2;
    Y3=deviation.*X3;
    h1=sum(Y1);
    h2=sum(Y2);
    h3=sum(Y3);
    H=[h1;h2;h3];
    theta=theta-alpha/m*H;
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
