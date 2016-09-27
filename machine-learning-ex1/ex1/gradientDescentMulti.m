function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
temp = zeros(n, num_iters);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    h = X*theta;
    temp(:,iter) = theta - ((alpha/m)*(X'*(h-y)));
    theta = temp(:,iter);
    
    %temp1=1/m*alpha*sum(X*theta-y);
    %temp2=1/m*alpha*sum((X*theta-y).*X(:,2));
    %temp3=1/m*alpha*sum((X*theta-y).*X(:,3));
    %theta(1)=theta(1)-temp1;
    %theta(2)=theta(2)-temp2;
    %theta(3)=theta(3)-temp3;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
