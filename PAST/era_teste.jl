using Plots
using ControlSystems
using LinearAlgebra
include("sysid.jl")
include("KugaPack.jl")


m = 1;
c = 0.5;
k = 200;

A = [
    0 1;
    -k/m -c/m
]

B = [
    0;
    1/m;
]

C = [1 0]

D = 0

sys = ss(A,B,C,D);

fs = 500
dt = 1/fs;
Tfinal = 20;

t = 0:dt:Tfinal;

y,t,x = impulse(sys,t);
#sysest = ERA_K(y,2,1,dt);

U = PulseGen(Tfinal/dt+1);
Y,T,X = lsim(sys,U,t);

sysarx = ARX_K(y,U,2,1,dt);

dampreport(sys)
dampreport(sysest)
dampreport(sysarx)