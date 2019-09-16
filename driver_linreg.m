%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Driver Linear Regression       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Linear Regression  %%
m = 2;
b = 3;
x = linspace(0,10,100)'; %Creates 100 numbers between 0 and 10(rows T columns)
y = m * x + b + 2*randn(100,1); %Distorts actual y solution
figure(1); clf; %Make central figure
plot(x,y,'rx'); %Plots distorted y against x 
hold on;
plot(x,m*x+b,'b-'); %Plots actual y against x

N = length(x);
f = @(p) norm(y - p(1) * x - p(2)) ^ 2 / N; %Loss function for linear refgression

f_grad = @(p) [-2 * sum(x .* (y - p(1) * x - p(2))) / N; ...
               -2 * sum(y - p(1) * x - p(2)) / N]; %Derivative of loss function
           
fdcheck(f, f_grad, [0;0]); %fdcheck derivative

output = steepest_descent(f, f_grad, [0;0], 10); %Search for m and b that
                                                  %minimize loss function

p = output(:,end); %Get m and b that minimize loss function

figure(1); %plot on same figure 1
plot(x,p(1)*x+p(2),'g-'); % Plot returned m and b that minimize loss function