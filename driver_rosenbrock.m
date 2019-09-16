%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            Driver Rosenbrock         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Rosenbrock %%
x0 = [0;0];
f = @(x)(1-x(1)).^2+100.*(x(2)-x(1).^2).^2;
gradf = @(x)[(-2+2.*x(1)-4*100.*x(1).*(x(2)-x(1).^2));
             100.*(2.*x(2)-2.*(x(1)).^2)];
         
figure(1), clf;
fdcheck(f, gradf, [2; 4]);

lr = 1;
figure(2), clf;
output = steepest_descent(f,gradf,x0,lr);

x = output(1, :);
y = output(1, :);

plot(x, y, 'x')
grid on
axis([-10 10 -10 10])
title("Rosenbrock Steepest Descent for n-variables")
xlabel("x - value");
ylabel("y - value");