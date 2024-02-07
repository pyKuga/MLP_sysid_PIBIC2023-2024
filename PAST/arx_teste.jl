#School of Mechanical Engineering - State University of Campinas
#Paulo Yoshio Kuga
#First Release: January 2022

#Present version: 20230715

#This is the main program that runs the analysis to reach out the ressearch results.

using LinearAlgebra
using ControlSystems
using Plots
using ControlSystemIdentification
using Random
using Statistics
using DelimitedFiles


include("KugaPack.jl")
include("sysid.jl")
include("blockpack.jl")
include("analysis.jl")


#numerical conditions are set up
dt = 1e-3;
maxT = 4;
ns = round(Int,maxT/dt)+1;
t = 0:dt:maxT;

#problem conditions are set up
x0 = [0; 0; 0; 0;];
U = PulseGen(ns);


#using the given parameters, the system (discrete space state) is created 
Param = [0.77 0.59 2.1 1.2 9 200 200 200];
sysP,sysD = F2DOF(dt,Param);

Y,_,_= lsim(sysD,U,t,x0=x0); #time response
magD,phaseD,w = bode(sysD); #frequency response

#Since F2DOF is a two degree freedom problem, we can state that the number of outputs is 2
#This implies the noise beign as (2,n)
#In this analysis, we are using MersenneTwister as the random number generator. 

seed = 98832+5556594 #seed for mersenne twister
noise = noise_gen(seed,2,ns);

fineza = 1;

NAmp = [i*1e-2 for i in 0:fineza:100];

#identification parameters
nx = 4
p = 700; #round(Int,ns/2);
na = 4
nb = 1


nsa, = size(NAmp);

NORMS = zeros(9,nsa);

#gerar a análise


ny,ns = size(Y); #determines the number of outputs and the number of samples
nu,_ = size(U); #determines the number of inputs
#if the signal or the input come as rows being the samples, it changes the order for columns being samples
Y,ny,ns = CheckData(Y,ny,ns); 
U,nu,ns = CheckData(U,nu,ns);

linhas = (ns-na)*ny; #since we want na coefficients to our denominator, we need to organize the samples.
colunas = na+nb*ny*nu; #it's related to the number of coefficients. 
H = zeros(linhas,colunas); #hankel matrix is reserved in memory

#for building our StateSpace system, we storage a matrix with the TF discrete type, to populate the array
G = Matrix{TransferFunction{Discrete{Float64}, ControlSystemsBase.SisoRational{Float64}}}(undef,ny,nu); 


#professor kurka formulation for MIMO-ARX with least square
Id = Matrix{Float64}(I,ny,ny);
H[:,1:nb*ny*nu] = kron(Hankel_PD2(U,nb,ns-na),Id);
H[:,(1:na).+nb*ny*nu] = -Hankel_PD(Y,na,linhas);
b = reshape(Y[:,na+1:ns],linhas,1);

Coef = H'*H\(H'*b);
A = reverse(push!(Coef[(1:na).+nb*ny],1));
tfA = tf(1,A,dt);

B = Coef[1:(nb*ny)];
 

#TF functions assembly and asssignment to the G matrix (the TF function one)
Bc = zeros(ny,nb+1); #a new matrix is called 
Bc[:,1:nb] = reverse(reshape(B,(ny*nu,nb)),dims=2); #it gets the coefficients and separe them into a nb row matrix, nu*ny columns. then the matrix is transposed, and reversed
for i=1:ny
    for j = 1:nu
        G[i,j] = 1/tf(1,Bc[i+j-1,:],dt)*tfA;
    end        
end



sysT = array2mimo(G); #conversion to a mimo system

Yout,_,_ = lsim(sysT,U,t);




