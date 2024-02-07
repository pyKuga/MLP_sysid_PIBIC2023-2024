using LinearAlgebra
using ControlSystems
using Plots
using ControlSystemIdentification
using Random
using Statistics
using DelimitedFiles
using FFTW


include("KugaPack.jl")
include("sysid.jl")
include("blockpack.jl")
include("analysis.jl")

#Constantes
g = 9.81;#[m/s^2]
l = 0.3302;#[m]

#Inercia
m1 = 0.38+0.37;#[kg]
m2 = 0.23;#[kg]
J0 = 7.88e-3;#[kg*m^3]

#Amortecimento
c1 = 5.4;#[Ns/m^2]
beta = 0.0024;#[N*m*s/rad]

#Rigidez
k = 142;#[N/m]

dt = 5e-2;
maxT = 100;
ns = round(Int,maxT/dt)+1;
t = 0:dt:maxT;
fs = 1/dt;


U_or = PulseGen(ns); 


fineza = 1;
MaxAmp = 100;
NAmp = [i for i in 0:fineza:MaxAmp].*1e-2;


nsa, = size(NAmp);
X0 = zeros(4,nsa)#1);
NORMS = zeros(9,nsa);

#identification parameters
nx = 4
p = 300; #round(Int,ns/2);
na = 4
nb = 1

Maxθ = pi/2


X0[2,:] = Maxθ*NAmp';

#--------------------------------------------------------ANALISE SOLTA------------------------------------------------
U = U_or

Yout,_,_ = RK4(X0[:,4],0.01*U,dt,maxT,ns);

plot(t,Yout',
label = ["x" "θ"], 
xlabel="Tempo (s)", 
ylabel="Deslocamento (m e rad)",
title = "Sistema para θ₀ = 2.7º")

sGRA = GRA(Yout,U,nx,dt,p);
sN4SID = n4sidPatternArg(Yout,U,dt,nx);
sARX = ARX_K(Yout,U,na,nb,dt);

YERA,_,_= lsim(sGRA,U,t);
YN4SID,_,_= lsim(sN4SID,U,t);
YARX,_,_= lsim(sARX,U,t);


plot(t,[Yout[1,:] ,YERA[1,:], YN4SID[1,:],YARX[1,:]], 
label = ["Original" "ERA" "N4SID" "ARX"],
xlabel="Tempo (s)", 
ylabel="Deslocamento (m)",
title = "Comparação: Resultado dos Algoritmos"
)

function PlotaGrafico(Y,legenda,channel,ns,fs,N)
    msim, psim, freq = FourierTransformAnalysis(Y,ns,fs); 
    y = 2*msim[channel,1001:2001];
    w = freq[freq .>= 0];
    arr = y;
    f(i,N,arr) = mean(arr[i-N:i]);
    h(i,N,arr) = mean(arr[i:i+N]);
    n(N,arr) = [[h(i,N,arr) for i=1:N]; [f(i,N,arr) for i=N+1:1001];]
    return plot(w,20*log.(n(N,arr)),label=legenda,legend = :outertopleft)
end

function PlotaGrafico!(Y,legenda,channel,ns,fs,N)
    msim, psim, freq = FourierTransformAnalysis(Y,ns,fs); 
    y = 2*msim[channel,1001:2001];
    w = freq[freq .>= 0];
    arr = y;
    f(i,N,arr) = mean(arr[i-N:i]);
    h(i,N,arr) = mean(arr[i:i+N]);
    n(N,arr) = [[h(i,N,arr) for i=1:N]; [f(i,N,arr) for i=N+1:1001];]
    return plot!(w,20*log.(n(N,arr)),label=legenda,legend = :outertopleft)
end

N = 5


PlotaGrafico(YERA,"ERA",1,ns,fs,N)
PlotaGrafico!(YN4SID,"N4SID",1,ns,fs,N)
PlotaGrafico!(YARX,"ARX",1,ns,fs,N)
PlotaGrafico!(Yout,"Original",1,ns,fs,1)


title!("FFT - Amplitude - Coordenada x")
xlabel!("Frequência (rad/s)")
ylabel!("Amplitude (dB)")


#--------------------------------------------------------ANALISES ROBUSTEZ------------------------------------------------


for i in 1:nsa
    U = U_or#*NAmp[i]
    Y,_,_ = RK4(X0[:,i],0.01*U,dt,maxT,ns);
    NORMS[1:3,i] = runAnalysis(GRA(Y,U,nx,dt,p),Y, U);
    NORMS[4:6,i] = runAnalysis(n4sidPatternArg(Y,U,dt,nx),Y, U);
    NORMS[7:9,i] = runAnalysis(ARX_K(Y,U,na,nb,dt),Y, U);
    print("|")
end

θ = rad2deg.(X0[2,:])

AVAL = NORMS;
tempo = plot(
    θ, 
    [AVAL[1,:] AVAL[4,:] AVAL[7,:]], 
    labels=["GRA (ERA)" "N4SID" "ARX"],
    ylabel = "ϵ", 
    xlabel = "Condição inicial em Graus (º)",
    title  = "ϵ com relação a Resposta Temporal",
);

amplitude = plot(
    θ, 
    [AVAL[2,:] AVAL[4,:] AVAL[8,:]], 
    labels=["GRA (ERA)" "N4SID" "ARX"],
    ylabel = "ϵ", 
    xlabel = "Condição inicial em Graus (º)",
    title  = "ϵ com relação a Amplitude (FFT)",
);


fase = plot(
    θ, 
    [AVAL[3,:] AVAL[5,:] AVAL[9,:]], 
    labels=["GRA (ERA)" "N4SID" "ARX"],
    ylabel = "ϵ", 
    xlabel = "Condição inicial em Graus (º)",
    title  = "ϵ com relação a Fase (FFT)",
);
png(amplitude,"amplitude_nl.png")
png(fase,"fase_nl.png")


plot(amplitude, fase, layout=(2,1))




Y,_,_ = RK4([0,0,0,0],U_or,dt,maxT,ns)

sistema = plot(
    t, 
    Y', 
    labels=["Posição do Carro" "Posição do Pêndulo"],
    ylabel = "Deslocamento (m) e \n Ângulo do Pêndulo (rad)", 
    xlabel = "Tempo (s)",                                                           
    title  = "Resposta para o sistema não-linear",
);

png(sistema,"pendulo.png")