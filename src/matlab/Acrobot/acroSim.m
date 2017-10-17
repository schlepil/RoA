function acroSim

p = AcrobotPlant;

% Set input limits
p = setInputLimits(p,-20,20);
p.b1 = 0.;
p.b2 = 0.;
p.g = 0.;

U = 0.;
X = [pi;0.1;10.;0.];


fInt = @(t,x) p.dynamics(t,x,U)
[Tsol, Xsol] = ode45(fInt, [0,3], X);

figure();hold all
plot(Tsol, Xsol(:,1:2))
figure();hold all
plot(Tsol, Xsol(:,3:4))

end







