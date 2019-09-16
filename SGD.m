%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Description: Computes stochastic gradient descent on a function, 
%              solving for what parameters(m, b) minimize the function
% Authors:       Akwarandu Ugo Nwachuku and Dr. Caleb Macgruder
% Date:          June 6th, 2019

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function output = SGD(f, gradf, batch_size, N, error_mode)

    parameters = [12; 4];
    xo = parameters;

    fprintf("      ********************** Verbose Mode ******************** \n")
    
    fprintf("    __________________________________________________________\n")
    fprintf("   |             Stochastic Gradient Descent Iterations       |\n")
    fprintf("   |----------------------------------------------------------|\n")
    fprintf("   |  i   |  f_value   |   grad_norm   |  bt_steps  |    lr   |\n")
    fprintf("   |----------------------------------------------------------|\n")
    
    
    
    lr = 2;
    beta = 0.5;
    index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    
    %Random indexes
    %Loop through in order through this dataset to get random non-repeat. 
    for i=1:N
        p(i) = randi(100);
    end
    
    iter = 0; 
    history = [];
    output_history = [];
    count = 1;
    %while norm(gradf(parameters, p(index))) > 1e-06 * norm(gradf(xo, p(index))) && iter * batch_size <= 500
    while count < 300
        count = count + 1;
        iter = iter + 1;
        lr = lr * 2; 
        
        %Distance travelled 
        grad_value = gradf(parameters, p(index));
        f_value = f(parameters, p(index));
        grad_norm = norm(grad_value);

        temp = parameters - lr * grad_value;
    
        steps = 0;
        
        while f(temp, p(index)) >= f_value && steps < 30
            lr = beta * lr;
            temp = parameters - lr * grad_value;
            steps = steps + 1;
        end
        output_history = horzcat(output_history, f(temp, p(index))); 
        parameters = temp;
        
  
        index = index + 10; 

        
        %Add to history 
        history = horzcat(history, parameters); 
        
        
        %Recycle
        if (index(length(index))) == N
            index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        end 
        
        fprintf("   |%3d   | %1.1e    |    %1.1e    |   %2d       | %1.1e |\n", iter, f_value, grad_norm, steps, lr)
        fprintf("   |----------------------------------------------------------|\n")
        
    if error_mode == true
        output = output_history;
    else
        output = history;
    end
end
