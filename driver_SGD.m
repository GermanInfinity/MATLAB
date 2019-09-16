%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             Driver SGD               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SGD function %

m = 2.5;
b = 1.71;
x = linspace(0,10,100)'; %Creates 100 numbers between 0 and 10(rows T columns)

opt = [m; b];
y = @(x_index) m * x(x_index) + b; %Y function
y_pred = @(opt, x_index) opt(1) * x(x_index) + opt(2); %+ 2*randn(100,1);%Distorted Y function

%Another function to test
%y = @(x_index) m * cos(x(x_index)) + b * sin(x(x_index)); %Y function
%y_pred = @(opt, x_index) opt(1) * cos(x(x_index)) + opt(2) * sin(x(x_index)); %+ 2*randn(100,1);%Distorted Y function

batch_size = 10;
p = [2,3];
N = 100;

%y_pred takes in two arguements. 
Q = @(init, idx) norm(y(idx) - y_pred(init, idx)) ^ 2 / N;
Q_grad = @(init, idx) [-2/N * sum(x(idx) .* (y(idx) - y_pred(init, idx))); ...
                       -2/N * sum(y(idx) - y_pred(init, idx))];
                   
Q_a = @(init) Q(init, [1, 5, 10, 15]);
Q_d = @(init) Q_grad(init, [1, 5, 10, 15]);

fdcheck(Q_a, Q_d, p, [1/(sqrt(2)), 1/(sqrt(2))])                  

% SGD 
print_error = true;
%output = SGD(Q, Q_grad, batch_size, N, print_error);




if print_error == true
    error_output = output;
    x = 1 : length(error_output);
    
    grid on
    figure(1); clf;
    %plot (x, error_output)
    semilogy (x, error_output)
    
    axis([0 length(error_output) -5 5])
    title("Error result from SGD")
    xlabel("Iteration");
    ylabel("Output error");

    
else
    x = 1 : 100;
    m = output(1, end);
    b = output(2, end);

    
    figure(1); clf;
    % y - actual
    plot(x, y(x), 'rx'); 
    grid on
    hold on
    % y - predicted
    plot (x, y_pred([m,b], x), 'b'); 

    axis([0 100 -40 40])
    title("SGD on loss function")
    xlabel("x - value");
    ylabel("y - value");
    legend('actual', 'predicted')
    
end