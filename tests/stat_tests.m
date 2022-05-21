clc;clear;close all;

X = [.6109,.5952,.6429]';
Y = [.665,.6755,.6378]';
Sp = (var(X)+var(Y))/2;

%  H0: X=Y (same var)
v = (mean(X)-mean(Y))/sqrt(Sp*(2/3));
p1 = tcdf(v,4)

% H0: var(x) = var(y)
v = var(X)/var(Y);
f = fcdf(v,2,2);
p2 = 2*min(f,1-f)


y = [X,Y];
p3 = anova1(y)

