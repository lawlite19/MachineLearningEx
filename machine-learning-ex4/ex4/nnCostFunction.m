function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));    %还原到初始的Theta1

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1); %行数
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_x = Theta1(:,(2:end));   %去掉theta1(0)
Theta2_x = Theta2(:,(2:end));   %去掉theta2(0)
regterm = [Theta1_x(:);Theta2_x(:)]'*[Theta1_x(:);Theta2_x(:)]; %theta的平方

class_y = zeros(m,num_labels);  %映射为0,1，主要后面求J中使用
for i = 1:num_labels
    class_y(:,i) = y==i;
end

%  正向传播
a1 = [ones(m,1),X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1),a2];
z3 = a2*Theta2';
h = sigmoid(z3);

%  求代价函数
J = -((class_y(:)'*(log(h(:)))) + ((1-class_y(:))'*(log(1-h(:))))-(lambda*regterm/2))/m;

%   反向传播求梯度
for i = 1:m
    delta3(i,:) = h(i,:)-class_y(i,:);  %误差率
    Theta2_grad = Theta2_grad+delta3(i,:)'*a2(i,:); %前一层，
    delta2(i,:) = (delta3(i,:)*Theta2_x).*sigmoidGradient(z2(i,:));
    Theta1_grad = Theta1_grad+delta2(i,:)'*a1(i,:);
end




    
Theta1(:,1) = 0;
Theta2(:,1) = 0;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = ([Theta1_grad(:);Theta2_grad(:)]+lambda*[Theta1(:);Theta2(:)])/m;


end
