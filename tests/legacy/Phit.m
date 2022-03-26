clc;clear;close all

data = [
0   0.24
50  0.25
100 0.25
150 0.25
200 0.26
250 0.26
300 0.27
350 0.27
400 0.28
450 0.29
500 0.30
550 0.31
600 0.32
700 0.36
800 0.43];

n = 4;

R = data(:,1);
sig = data(:,2);
% p = polyfit(R,sig,n);
p = [9.5338e-13 -9.2819e-10 3.9214e-07 2.4187e-05 .24316];
R2 = linspace(0,900);
plot(R,sig,'.-',R2,polyval(p,R2),'linewidth',2,'markersize',24)
legend('Data','Polynomial approximation','location','best')
set(gca,'fontsize',12)
grid on
xlabel('Range [m]')
ylabel('std [mils]','fontsize',13)
saveas(gcf,'~/Documents/ERM/2Ma/thesis/images/std(range).png')

figure
R = linspace(0,3,1e3);
phit = @(r) erf(1./polyval(p,r)/sqrt(2));

% phit = @(r) 1-2*normcdf(-1./polyval(p,r));
% phit = @(r) chi2cdf(1./polyval(p,r),2);
f = @(x) phit(x) - 0.5;
R50 = fsolve(f,1000)
Ph = @(R) phit(R50*R);
% R2 = linspace(.5,3);
% p = polyfit(R,Ph(R),16);
% Ph2 = @(R) (R<.5) + polyval(p,R).*(R<3 & R>=.5);
plot(R,Ph(R),[-1,1,1],[Ph(1),Ph(1),-1],'o--','linewidth',2)
xlim([R(1) R(end)])
ylim([0 1])
set(gca,'fontsize',12)
grid on
xlabel('Normalized range (R/R50)')
ylabel('P_{kill}','fontsize',13)
saveas(gcf,'~/Documents/ERM/2Ma/thesis/images/pkill.png')

% clc;clear;close all;
% 
% R50 = 1;
% n = 7.5;
% a = 10;
% 
% r = linspace(0,2*R50)';
% 
% % sig = exp((r/R10).^4*log(.1));
% Ph = @(r) 1./(1+(r/R50).^n);
% % sig = 1-gammainc(10*r/R50,a);
% 
% % figure('windowstate','maximized')
% plot(r,Ph(r),[-1,R50,R50],[Ph(R50),Ph(R50),-1],'o--','linewidth',2)
% xlim([r(1) r(end)])
% ylim([0 1])
% set(gca,'fontsize',12)
% grid on
% xlabel('Normalized range (R/R50)')
% ylabel('P_{hit}','fontsize',13)
% saveas(gcf,'~/Documents/ERM/2Ma/thesis/images/phit.png')
