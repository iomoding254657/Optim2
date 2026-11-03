%% Hollow drive shaft optimization: compare six MATLAB optimizers
% Save this file as shaft_optimizer_comparison.m and run it in MATLAB.
% Requires Optimization Toolbox and Global Optimization Toolbox.
% All dimensions use SI units: metres, N*m, Pa, kg.

clear; clc; close all;
rng(7,'twister');                 % reproducible starting random stream

% Engineering data
L = 0.80;                         % shaft length (m)
T = 500;                          % transmitted torque (N*m)
rho = 7850;                       % steel density (kg/m^3)
G = 80e9;                         % steel shear modulus (Pa)
tauAllow = 80e6;                  % allowable shear stress (Pa)
thetaAllow = 0.020;               % maximum twist (rad)

% Design variables: x(1) = outside diameter D, x(2) = wall thickness t
lb = [0.040, 0.002];
ub = [0.080, 0.015];
x0 = [0.060, 0.008];
nvars = 2;

% Exact nonlinear constraints are g(x) <= 0.
nonlcon = @shaftConstraints;

% For bound-only solvers (particleswarm and simulannealbnd), use a
% quadratic exterior penalty for the same normalized constraints.
penaltyFun = @(x) shaftMass(x) + 1e6*sum(max(shaftIneq(x),0).^2);

% Repeat stochastic solvers so timings and solutions are less seed-dependent.
% Median time is reported. Set repeats=1 for a quicker first run.
repeats = 5;
solverNames = {'ga','fmincon','particleswarm','patternsearch', ...
               'surrogateopt','simulannealbnd'};
times = zeros(numel(solverNames),repeats);
masses = nan(numel(solverNames),repeats);
solutions = nan(numel(solverNames),2,repeats);

for r = 1:repeats
    rng(100+r,'twister');

    % 1) Genetic algorithm: nonlinear constraints passed directly.
    opts = optimoptions('ga','Display','off','PopulationSize',60, ...
        'MaxGenerations',150,'FunctionTolerance',1e-8);
    tic;
    [x,fval] = ga(@shaftMass,nvars,[],[],[],[],lb,ub,nonlcon,opts);
    times(1,r) = toc;
    solutions(1,:,r) = x; masses(1,r) = fval;

    % 2) fmincon: gradient-based constrained local solver.
    opts = optimoptions('fmincon','Algorithm','sqp','Display','off', ...
        'MaxIterations',500,'OptimalityTolerance',1e-9, ...
        'ConstraintTolerance',1e-8);
    tic;
    [x,fval] = fmincon(@shaftMass,x0,[],[],[],[],lb,ub,nonlcon,opts);
    times(2,r) = toc;
    solutions(2,:,r) = x; masses(2,r) = fval;

    % 3) Particle swarm: penalty objective because particleswarm accepts bounds.
    opts = optimoptions('particleswarm','Display','off', ...
        'SwarmSize',60,'MaxIterations',200,'FunctionTolerance',1e-8);
    tic;
    [x,~] = particleswarm(penaltyFun,nvars,lb,ub,opts);
    times(3,r) = toc;
    solutions(3,:,r) = x; masses(3,r) = shaftMass(x);

    % 4) Pattern search: nonlinear constraints passed directly.
    opts = optimoptions('patternsearch','Display','off', ...
        'MaxIterations',1000,'FunctionTolerance',1e-8, ...
        'ConstraintTolerance',1e-8);
    tic;
    [x,fval] = patternsearch(@shaftMass,x0,[],[],[],[],lb,ub,nonlcon,opts);
    times(4,r) = toc;
    solutions(4,:,r) = x; masses(4,r) = fval;

    % 5) Surrogate optimization: its objective callback returns objective + Ineq.
    opts = optimoptions('surrogateopt','Display','off', ...
        'MaxFunctionEvaluations',500,'MinSurrogatePoints',30);
    tic;
    [x,~] = surrogateopt(@shaftSurrogate,lb,ub,[],[],[],[],[],opts);
    times(5,r) = toc;
    solutions(5,:,r) = x; masses(5,r) = shaftMass(x);

    % 6) Simulated annealing: bound constraints plus the same penalty.
    opts = optimoptions('simulannealbnd','Display','off', ...
        'MaxIterations',3000,'FunctionTolerance',1e-8);
    tic;
    [x,~] = simulannealbnd(penaltyFun,x0,lb,ub,opts);
    times(6,r) = toc;
    solutions(6,:,r) = x; masses(6,r) = shaftMass(x);
end

% Summarize using median runtime and median feasible mass across repeats.
medianTime = median(times,2);
medianMass = median(masses,2);
Dresult = median(solutions(:,1,:),3);
tresult = median(solutions(:,2,:),3);
feasible = false(numel(solverNames),1);
for k = 1:numel(solverNames)
    feasible(k) = all(shaftIneq([Dresult(k),tresult(k)]) <= 1e-5);
end
Results = table(solverNames',medianTime,medianMass,Dresult,tresult,feasible, ...
    'VariableNames',{'Optimizer','MedianTime_s','MedianMass_kg', ...
    'OuterDiameter_m','WallThickness_m','Feasible'});
disp(Results);

% Runtime comparison graph
figure('Color','w');
bar(categorical(solverNames),medianTime);
ylabel('Median computation time (s)');
title(sprintf('Runtime comparison for hollow shaft optimization (%d runs)',repeats));
grid on;

% Optional mass comparison graph
figure('Color','w');
bar(categorical(solverNames),medianMass);
ylabel('Median shaft mass (kg)');
title('Best design quality comparison');
grid on;

% ---- Local functions ----
function m = shaftMass(x)
    D = x(1); t = x(2);
    di = D - 2*t;
    m = 7850*pi/4*(D^2-di^2)*0.80;
end

function g = shaftIneq(x)
    D = x(1); t = x(2);
    di = D - 2*t;
    J = pi/32*(D^4-di^4);
    tau = 500*(D/2)/J;
    theta = 500*0.80/(J*80e9);
    g = [tau/80e6 - 1, theta/0.020 - 1];
end

function [c,ceq] = shaftConstraints(x)
    c = shaftIneq(x);             % MATLAB convention: c <= 0
    ceq = [];
end

function out = shaftSurrogate(x)
    out.Objective = shaftMass(x);
    out.Ineq = shaftIneq(x);      % surrogateopt constraint convention
end
