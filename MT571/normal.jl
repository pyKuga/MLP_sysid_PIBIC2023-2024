using LinearAlgebra
using ControlSystems
using Plots
using Random
using Statistics

include("ARX_ON.jl")

m = 1
c = 2
k = 50

A = [
    0 1;
    -k/m -c/m
]

B = [0 ; 1/m]

C = [1 0]

D = 0

sys = ss(A,B,C,D)

#numerical conditions are set up
dt = 2e-5;
maxT = 6;
ns = round(Int,maxT/dt)+1;
t = 0:dt:maxT;

#problem conditions are set up
x0 = [0; 0];

#identification parameters
U = ones(1,ns);#PulseGen(ns); 
Y,_,_ = lsim(sys,U,t,x0);


na= 2
nb = 1
η = (1:10:1000)*1e-4
tol = 1e-5


sysARX,or_Coef,H,b = ARX_K(Y,U,na,nb,dt);
Y_or,_,_ = lsim(sysARX,U,t,x0)

θ = [or_Coef[2],or_Coef[3],or_Coef[1]]

plot(
    t, 
    [Y' Y_or'],
    labels=["Sinal Original" "ARX"],
    ylabel = "Posição (m)", 
    xlabel="Tempo (s)"
)

Arduino = ControllerInit(0.01,na,nb,tol);
grad_Bulid(Arduino,ns,Y,U)

norma = []
for ratio in η
    Arduino.θ = rand(na+nb)
    Arduino.η = ratio
    GDS(Arduino)
    norma = norm(θ-Arduino.θ)/norm(θ)
end



AdamRun(Arduino)
Adamcoef = Arduino.θ


# Arduino.M\Arduino.L

# Identified = tf(Arduino.θ[na+nb],[1; -Arduino.θ[1:na]],dt)
# dampreport(Identified)


# Ys,_,_ = lsim(Identified,U,t,x0);
# plot(t,Ys')


