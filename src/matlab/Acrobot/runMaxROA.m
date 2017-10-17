function runMaxROA

p = AcrobotPlant;

% Set input limits
p = setInputLimits(p,-20,20);

% Do lqr
Q = diag([10 10 1 1]);
R=0.1;

x0 = [pi;0;0;0];
x0A = [pi;0;0;0];
u0 = 0;
[c,V] = tilqr(p,x0,u0,Q,R);

% Do Taylor expansions
x0 = Point(p.getStateFrame,[pi;0;0;0]);
u0 = Point(p.getInputFrame,0);
sys = taylorApprox(p,0,x0,u0,3);

% Options for control design
options = struct();
options.controller_deg = 1%1; % Degree of polynomial controller to search for
options.max_iterations = 3; % Maximum number of iterations (3 steps per iteration)
options.converged_tol = 1e-3; % Tolerance for checking convergence

options.rho0 = 0.01; % Initial guess for rho

options.clean_tol = 1e-6; % tolerance for cleaning small terms
options.backoff_percent = 5; % 1 percent backing off

options.degL1 = options.controller_deg + 1; % Chosen to do degree matching
options.degLu = options.controller_deg - 1; 
options.degLu1 = 2; 
options.degLu2 = 2;
options.degLup = 2;
options.degLum = 2; 

% Perform controller design and verification
disp('Starting verification...')
[c,V] = maxROAFeedback(sys,c,V,options);

% V = V.inFrame(sys.getStateFrame());

% Test stuff
% Create closed loop system with optimized controller
sysCl = feedback(p,c);
V0 = V.getPoly;
xinit = getLevelSet(decomp(V0),V0,struct('x0',[0;0;0;0]));

dlmwrite('Pout.txt', V.S);
ff = fopen('KoutStr.txt', 'rt');
Kstr = fgetl(ff);
fclose(ff)
Kstr = regexprep(Kstr, 'x1', 'x(1)');
Kstr = regexprep(Kstr, 'x2', 'x(2)');
Kstr = regexprep(Kstr, 'x3', 'x(3)');
Kstr = regexprep(Kstr, 'x4', 'x(4)');

eval(['K=@(x)', Kstr]);

% Create closed loop system with optimized controller
V0 = V.getPoly;
xinit = 0.95*getLevelSet(decomp(V0),V0,struct('x0',[0;0;0;0]));

if 0
	%Do some simulations
	fInt = @(t,x) p.dynamics(t, x, max(min(K(x-x0A),20),-20));
	figure(); hold all;
	for k = 1:size(xinit,2)
		  [tSol, xSol] = ode45(fInt, [0,2], xinit(:,k)+x0A);
		  plot(xSol(:,1), xSol(:,2))
		  dlmwrite(['X',num2str(k), '.txt'], xSol )
		  dlmwrite(['T',num2str(k), '.txt'], tSol )
	end
end
end






