%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Verify 1-layer Derivatives     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
x = linspace(0,10,100)';
A = 2.5; %m
b = 1.71;

%SIGMOID FUNCTION 
sigmoid = @(x)1./(1+exp(-x));
sigmoid_deriv = @(x) (exp(-x))./((1+exp(-x)).^2);

opt = [A; b];
y = @(x_index) sigmoid(A * x(x_index) + b);
y_pred = @(opt, x_index) sigmoid(opt(1) * x(x_index) + opt(2));
p = [2,3]; %Guess
N = 100;

Q = @(init, idx) norm(y(idx) - y_pred(init, idx)) ^ 2 / N;
Q_grad = @(init, idx) [-2/N * sum((y(idx) - y_pred(init, idx)) .* sigmoid_deriv(A * x(idx) + b) .* A .* x(idx)); ...
                       -2/N * sum((y(idx) - y_pred(init, idx)) .* sigmoid_deriv(A * x(idx) + b))];
                   
                   
Q_a = @(init) Q(init, [1, 5, 10, 15]);
Q_d = @(init) Q_grad(init, [1, 5, 10, 15]);

fdcheck(Q_a, Q_d, p, [1/(sqrt(2)), 1/(sqrt(2))])   