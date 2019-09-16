%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             Driver Quad              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Quad function %%
x0 = [-5;10];
f = @(x) (0.5 .* x(1).^2 + 0.5/10  .*  x(2).^2) + 1;
gradf = @(x)[x(1); x(2)/10];

figure(1), clf;
fdcheck(f, gradf, x0);

lr = 1;
figure(2), clf;

output = steepest_descent(f,gradf,x0,lr);

x = output(1, :);
y = output(1, :);

plot(x, y, 'x')
grid on
axis([-10 10 -10 10])
title("Steepest Descent for n-variables")
xlabel("x - value");
ylabel("y - value");