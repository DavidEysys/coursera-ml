function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%Theta1 (25x401), Theta2 (10x26), X (5000x400), p (5000x1)

%Layer 1 (Input)

a1 = [ones(m, 1), X]; % add 1 for the bias (ones(m, 1) gives us a column vector of ones)
% a1 (5000x401)


% a2 = g(z^(2)), z^(2) = Theta^(1) * a1
% a1 is (5000x401) and Theta1 is (25x401) ==> Theta1 * a1' (25x5000)

z2 = Theta1 * a1';	% z2 (25x5000)

%Layer 2 (Hidden layer)

a2 = sigmoid(z2); % a2 (25, 5000)

% Have to add bias (want a (26x5000), so opposite of before)

a2 = [ones(1, size(a2, 2)); a2];	% a2 (26x5000)

% Layer 3 (Output)

% Theta2 (10x26), a2 (26x5000) ==> Theta2 * a2 (10x5000)

z3 = Theta2 * a2;	% z3 (10x5000)

a3 = sigmoid(z3);	% a3 (10x5000), This is our hypothesis h

% Note, h should have dimensions (m, num_labels) and we have one hypothesis per training example such that the vector h(i,:) classifies our example

[x, p] = max(a3', [], 2);	% Transpose a3 so p is a 5000x1 vector



% =========================================================================


end
