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
U = 0.01*PulseGen(ns); 

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
nx = 4;
p = 700; #round(Int,ns/2);
na = 4;
nb = 1;


nsa, = size(NAmp);

NORMS = zeros(9,nsa);

#--------------------------------------------------------ANALISE SOLTA------------------------------------------------


Yout = Y+NAmp[3]*noise.*maximum(Y,dims=2);
sGRA = GRA(Yout,U,nx,dt,p);
sN4SID = n4sidPatternArg(Yout,U,dt,nx);
sARX = ARX_K(Yout,U,na,nb,dt);

YERA,_,_= lsim(sGRA,U,t);
YN4SID,_,_= lsim(sN4SID,U,t);
YARX,_,_= lsim(sARX,U,t);

plot(t,Yout', 
    label = ["x1 com Ruído" "x2 com Ruído"], 
    xlabel="Tempo (s)", 
    ylabel="Deslocamento (m)",
    title = "Saída com Ruído"
    )

plot(t,[Y[1,:] ,YERA[1,:], YN4SID[1,:],YARX[1,:]], 
label = ["Original" "ERA" "N4SID" "ARX"],
xlabel="Tempo (s)", 
ylabel="Deslocamento (m)",
title = "Comparação: Resultado dos Algoritmos"
)

function PlotaGrafico(sysT,legenda,w,channel)
    msim, psim, = bode(sysT,w);
    msim = reshape(msim,(2,240));
    psim = reshape(psim,(2,240));
    y = msim[channel,:];
    return plot(w,20*log.(y),xaxis=:log,label=legenda,legend = :outertopleft)
end

function PlotaGrafico!(sysT,legenda,w,channel)
    msim, psim, = bode(sysT,w);
    msim = reshape(msim,(2,240));
    psim = reshape(psim,(2,240));
    y = msim[channel,:];
    return plot!(w,20*log.(y),xaxis=:log,label=legenda,legend = :outertopleft)
end

PlotaGrafico(sGRA,"ERA",w,1)
PlotaGrafico!(sN4SID,"N4SID",w,1)
PlotaGrafico!(sARX,"ARX",w,1)
#PlotaGrafico!(sysD,"Original",w,1)

magDts = reshape(magD,(2,240));
inp = magDts[1,:]
plot!(w,20*log.(inp),
xaxis=:log,
label="Original",
legend = :outertopleft,
title="Diagrama de Bode - Coordenada x₁",
xlabel="Frequência (rad/s)",
ylabel="Amplitude (dB)")
 

#gerar a análise
#--------------------------------------------------------ANALISES ROBUSTEZ------------------------------------------------

for i in 1:nsa
    ruido = NAmp[i]*noise.*maximum(Y,dims=2);
    Yout = Y+ruido

    NORMS[1:3,i] = runAnalysis(GRA(Yout,U,nx,dt,p),Y, U,magD, phaseD,t);
    NORMS[4:6,i] = runAnalysis(n4sidPatternArg(Yout,U,dt,nx),Y, U,magD, phaseD,t);
    NORMS[7:9,i] = runAnalysis(ARX_K(Yout,U,na,nb,dt),Y, U ,magD, phaseD,t);
    print("*")

end



# U = zeros(1,ns);
# U[1] = 1;
# Y,_,_= lsim(sysD,U,t,x0=x0); #time response

# NORMS1 = zeros(9,nsa);

# for i in 2:nsa
#     ruido = NAmp[i]*noise.*maximum(Y,dims=2);
#     Yout = Y+ruido

#     NORMS1[1:3,i] = runAnalysis(GRA(Yout,U,nx,dt,p),Y, U,magD, phaseD);
#     NORMS1[4:6,i] = runAnalysis(n4sidPatternArg(Yout,U,dt,nx),Y, U,magD, phaseD);
#     NORMS1[7:9,i] = runAnalysis(ARX_K(Yout,U,na,nb,dt),Y, U,magD, phaseD);

# end


AVAL = NORMS;
plot(
    NAmp, 
    [AVAL[1,:] AVAL[3,:] AVAL[7,:]], 
    labels=["GRA (ERA)" "N4SID" "ARX"],
    ylabel = "ϵ", 
    xlabel = "Percentual de Amplificação do ruído",                                                           
    title  = "ϵ com relação a Resposta Temporal",
)

amplitude = plot(
    NAmp, 
    [AVAL[2,:] AVAL[4,:] AVAL[8,:]], 
    labels=["GRA (ERA)" "N4SID" "ARX"],
    ylabel = "ϵ", 
    xlabel = "Percentual de Amplificação do ruído",                                                           
    title  = "ϵ com relação a Amplitude (Transformada de Bode)",
);



fase = plot(
    NAmp, 
    [AVAL[3,:] AVAL[5,:] AVAL[9,:]], 
    labels=["GRA (ERA)" "N4SID" "ARX"],
    ylabel = "ϵ", 
    xlabel = "Percentual de Amplificação do ruído",                                                           
    title  = "ϵ com relação a Fase (Transformada de Bode)",
);


png(amplitude,"amplitude_l.png")
png(fase,"fase_l.png")

#plot(amplitude, fase, layout=(2,1))



# plot(
#     t, 
#     Y', 
#     labels=["Posição do Bloco 1" "Posição do Bloco 2"],
#     ylabel = "Deslocamento (m)", 
#     xlabel = "Tempo (s)",                                                           
#     title  = "Resposta para o sistema linear",
# )
