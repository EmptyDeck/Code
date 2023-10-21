close all;
clear all;
clc;

n = [-10:1:10];

% Delta function
xa = (n == 0);

% Step function shifted by -5 units
xb = stepfun(n,-5)

% Rectangle pulse: Assuming width of 9 and centered at 0
xc = recpuls(n/9)

% Ramp signal: Assuming slope of 0.2 and shifted by 3 units
xd = 0.2 * (n - 3) .* (n >= 3);

xe = sinc(n/3)

xf = exp(-0.15*n).*cos(0.2*pi*n).*(n>=0);

% Plotting
subplot(3,2,1);
stem(n,xa);
axis([-10 10 -0.5 1.5]); grid on;
title('Input signal \delta[n]');

subplot(3,2,2);
stem(n,xb);
axis([-10 10 -0.5 1.5]); grid on;
title('Step function u[n+5]');

subplot(3,2,3);
stem(n,xc);
axis([-10 10 -0.5 1.5]); grid on;
title('Square pulse rect[n/9]');

subplot(3,2,4);
stem(n,xd);
axis([-10 10 -0.5 4]); grid on;
title('Ramp signal 0.2r[n-3]');


subplot(3,2,5);
stem(n,xd);
axis([-10 10 -0.5 4]); grid on;
title('Sinc function');

subplot(3,2,6);
stem(n,xf);
axis([-10 10 -1 1]); grid on;
title('지수감쇠 정현파');
