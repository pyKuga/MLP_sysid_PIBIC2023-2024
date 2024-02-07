#School of Mechanical Engineering - State University of Campinas
#Paulo Yoshio Kuga
#First Release: October 2023

#Present version: 20231001

#this is the main program to run neural network identification 

using LinearAlgebra
using ControlSystems
using Plots
using ControlSystemIdentification
using Random
using Statistics
using DelimitedFiles
using CUDA
using Flux
using Hyperopt
using FluxArchitectures


include("KugaPack.jl")
include("sysid.jl")
include("blockpack.jl")
include("analysis.jl")


#numerical conditions are set up
dt = 1e-3;
maxT = 10;
ns = round(Int,maxT/dt)+1;
t = 0:dt:maxT;

#problem conditions are set up
x0 = [0; 0; 0; 0;];

w = 5
U = 4*rand(1,ns);
#ones(1,ns)'
#chirp(w,t)';
#0.01*PulseGen(ns); 

#using the given parameters, the system (discrete space state) is created 
Param = [0.77 0.59 2.1 1.2 9 200 200 200];
sysP,sysD = F2DOF(dt,Param);

Y,_,_= lsim(sysD,U,t,x0=x0); #time response
magD,phaseD,w = bode(sysD); #frequency response

#plot(t,Y')

#Since F2DOF is a two degree freedom problem, we can state that the number of outputs is 2
#This implies the noise beign as (2,n)
#In this analysis, we are using MersenneTwister as the random number generator. 

seed = 98832+5556594 #seed for mersenne twister
noise = noise_gen(seed,2,ns);

fineza = 1;

NAmp = [i*1e-2 for i in 0:fineza:100];

#identification parameters
nx = 4

na = 100;
nb = 1;
ny = 2;
nu = 1;
p = 25

V = [nu, 4, 4, 4, ny]

model = Simple(V,tanh) |> gpu
Data = SimpleData(Y,U,ns)

opt_state = Flux.setup(Adam(), model); 
Flux.train!(model, Data, opt_state) do m, x, y
        Flux.mse(m(x), y);
end

Yr = [model(i[1]) for i in Data]

Yl = zeros(ny,ns) |> gpu

for i=1:ns
        Yl[:,i] = Yr[i]
end

in, convlayersize, recurlayersize, poolsize, skiplength = (1,3,4,120,1)

LSTnet(inputsize, convlayersize, recurlayersize, poollength, skiplength,
    init=Flux.zeros32, initW=Flux.zeros32) |> gpu