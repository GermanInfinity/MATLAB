%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Description: Computes steepest descent on a function of n variables 
% Authors:       Akwarandu Ugo Nwachuku and Dr. Caleb Macgruder
% Date:          May 29th, 2019

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function output = steepest_descent(f,gradf,x0,lr)
 
    beta = 0.5;
    history = [];
    opt = x0;
    history = horzcat(history, opt);

    fprintf("      ********************** Verbose Mode ******************** \n")
    
    fprintf("    __________________________________________________________\n")
    fprintf("   |               Steepest Descent Iterations                |\n")
    fprintf("   |----------------------------------------------------------|\n")
    fprintf("   |  i   |  f_value   |   grad_norm   |  bt_steps  |    lr   |\n")
    fprintf("   |----------------------------------------------------------|\n")
    
    iter = 0;
    
    while norm(gradf(opt)) > 1e-06 * norm(gradf(x0)) && iter < 100
        
        iter = iter + 1;
        %Distance travelled
        grad = gradf(opt);
        opt = opt - lr * grad;
        
        %Add to history 
        history = horzcat(history, opt);
        
        f_value = f(opt);
        norm_grad = norm(gradf(opt));
        
        %lr = 15;
        grad = gradf(opt);
        steps = 0;
        
        %Backtracking
        %Create temp for next step in backtracking search
        temp = opt - lr * grad;   
        armijo = - 1e-02 * lr * (norm_grad ^ 2);
        
        %Evaluate new function value
        while f(temp) >= f_value + armijo && steps < 30
            armijo = - 1e-02 * lr * (norm_grad ^ 2);
            lr = beta * lr;
            temp = opt - lr * grad;
            steps = steps + 1;
        end
        
        fprintf("   |%3d   | %1.1e    |    %1.1e    |   %2d       | %1.1e |\n", iter, f_value, norm_grad, steps, lr)
        fprintf("   |----------------------------------------------------------|\n")
        
        
    end
    fprintf("    *************************End************************** \n")
    
    output = history;
end
