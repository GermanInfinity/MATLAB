%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Description: Computes finite difference between functions of n variables
%              and plots the error. 
% Authors:       Akwarandu Ugo Nwachuku and Dr. Caleb Macgruder
% Date:          May 18th, 2019

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [min_error] = fdcheck(f, fp, x, p)
    
    if nargin < 4
        p = randn(length(x), 1);
    end

    eps = logspace(-9, -1, 100);
    
    count = 0;
    error = zeros(1, 10);
    
    for epsilon = logspace(-9, -1, 100)
        f_approx = (f(x + epsilon .* p) - f(x - epsilon .* p)) / (2 * epsilon);
        f_grad = dot(fp(x), p);
        count = count + 1;
        error(count) = norm(f_approx - f_grad); 
 
    end
    figure;
    loglog(eps, error);
    xlabel('Epsilon, e');
    ylabel('Absolute Error');
    title('Error in FD Approximation, [ord=2]');
       

end
