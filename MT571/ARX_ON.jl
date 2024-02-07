#MT571 - Main Report

#Paulo Yoshio Kuga - 204451

#auxiliary library that contains code for the controller 

#the structure for the controller 
mutable struct Controller
    M::Matrix
    L::Vector
    h::Vector
    θ::Vector
    ∇::Vector
    na::Int64  
    nb::Int64
    i::Int64
    η::Float64
    tol::Float64
    m::Int64
end

#adam is a separated structure due to it complexity
mutable struct AdamOpt
    η::Float64
    α::Float64
    β::Float64
    t::Int64
    ϵ::Float64
    m::Vector
    v::Vector
end

#initializes the controller
function ControllerInit(η,na,nb,ns,tol)
    z = zeros(na+nb);
    r = rand(na+nb)
    i = na
    return Controller(z*z',z,z,r,r,na,nb,i,η,tol,ns-na)
end

#runs the RMS function to the controller 
function  MSEGrad(c)
    return 2*(c.M*c.θ-c.L)/c.m
end


#initializes ADAM 
function AdamInit(η,α,β,ϵ)
    t = 0
    z = zeros(na+nb)
    AdamOpt(η,α,β,t,ϵ,z,z)
end

#runs ADAM with controller parameters
function AdamRun(c)
    opt = AdamInit(c.η,0.9,0.999,1e-8)
    Δ = 100
    while norm(Δ)/norm(c.θ) > c.tol
        opt.t += 1;
        c.∇ = MSEGrad(c);
        opt.m = opt.α*opt.m + (1-opt.α)*c.∇;
        opt.v = opt.β*opt.v + (1-opt.β)*c.∇.^2;
        hat_m = opt.m/(1-opt.α^opt.t);
        hat_v = opt.v/(1-opt.β^opt.t);
        Δ = c.η*hat_m./(sqrt.(hat_v).+opt.ϵ);
        c.θ = c.θ - Δ;
    end
end

#bulids the auxiliary matrices in the controller 
function grad_Bulid(Arduino,ns,Y,U)
    h = Arduino.h;
    M = Arduino.M;
    L = Arduino.L;
    i = Arduino.i;

    while i < ns
        h[1:na] = Y[(1:na).+(i-na)];
        h[na+nb] = U[i+1]; #apenas para o caso onde nb =1
        M .+= h*h'; #aqui é o inverso da formulação, pois vetores no julia são verticais
        L .+= h*Y[i+1] ;
        i += 1
    end
end


#procedes with Gradient Descent Method on controller memory
function GDS(c)
    while norm(c.η*c.∇)/norm(c.θ) > c.tol
        c.∇ = MSEGrad(c);
        c.θ = c.θ - c.η*c.∇;
    end
end


#check if signal is at the right side
function CheckData(Y,ny,ns)
    if ns < ny
        Y = Y';
        ny,ns = size(Y);
    end
    return Y, ny, ns
end


#THIS EXCERPT OF CODE WAS TAKEN FROM MY ORIGINAL LIBRARY SYSID.JL - https://github.com/pyKuga/MT571/blob/main/sysid.jl

#Hankel pré-determinado
#np ordem do polinimio
#ns numero de amostras
function Hankel_PD(Y,np,linhas)
    H = zeros(linhas,np); #estabelecer matriz de hankel baseada no número de linhas e entradas desejadas
    ny,ns = size(Y);
    Yf = reshape(Y,(ny*ns,1));
    for i in 1:np
        H[:,i] = Yf[(1:linhas).+(i-1)*ny]#reshape(Y[:,(1:(ns-np)).+(i-1)],linhas,1);
    end
    return H
end

function Hankel_PD2(U,np,lines)
    nu,ns = size(U);
    Ut = reshape(U,(1,ns*nu));
    F = zeros(lines,np*nu); #estabelecer matriz de hankel baseada no número de linhas e entradas desejadas
    for i in 1:lines 
        F[i,:] = Ut[(1:np*nu).+(i-1)*nu];
    end
    return F
end

#Y: multiple output signal
#U: multiple input signal
#na: denominator polynomial order na>=nb
#nb: numerator polynomial order nb>=1
function ARX_K(Y,U, na, nb,dt)
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

    

    return array2mimo(G),Coef, H,b; #conversion to a mimo system
    #returns the space state form of the system, not the TF.
end

