using Flux
using CUDA


#School of Mechanical Engineering - State University of Campinas
#Paulo Yoshio Kuga

#This package is beign written for "Análise de Robustez em Métodos de Identificação de Sistemas", a undergraduate ressearch for PIBIC.
#This package is also being used in "Análise de Robustez em Métodos de Identificação de Sistemas e Redes Neurais" the second signed PIBIC propposal.

#Main content: sysid algorithms and it's support functions

#------------------------------------------GENERAL FUNCTIONS-------------------------------------------------

function CheckData(Y,ny,ns)
    if ns < ny
        Y = Y';
        ny,ns = size(Y);
    end
    return Y, ny, ns
end

#------------------------------------------ERA RELATED FUNCTIONS-------------------------------------------------

function Hankel(h,p,ny)
    h = h[:,2:p+1];
    ch = round(Int,p/2); #colunas da matriz de hankel
    rh = ch; #linhas da matriz de hankel
    H = zeros(rh*ny,ch+1); #estabelecer matriz de hankel baseada no número de linhas e entradas desejadas
    for i in 1:rh #preenche a matriz de hankel (dificil explicar a logica, mas em suma tenta tomar os espaços já pré-determinados para preencher a matriz)
        H[(1:ny).+((i-1)*ny),:] = h[:,(1:ch+1).+(i-1)]
    end
    H1 = H[:,1:ch];
    H2 = H[:,(1:ch).+1];
    return H1, H2
end

function ERA_K(Y,nx,nu,dt,p,ny)
    H1,H2 = Hankel(Y,p,ny);
    U, S, V = svd(H1); #faz a SVD
    S = diagm(S[1:nx]); #transforma os autovalores do svd em uma matriz diagonal
    U = U[:,1:nx]; #toma as colunas relativas a ordem do modelo
    V = V[:,1:nx]; #confirmar com o prof a hipótese do V ser tomado como coluna devido a transposição e da definição do SVD em si
    G = S.^0.5; #tira a raiz quadrada dos autovalores
    Q = inv(G); #inverte-os
    Or = U*G; #observabilidade do modelo
    Cr = G*V'; #controlabilidade do modelo
    A = Q*U'*H2*V*Q; #gera a matriz A
    C = Or[1:ny,:]; #com base no número de entradas do modelo, gera o 
    B = Cr[:,1:nu]; #numero de entradas é um dado do problema
    D = Y[:,1];
    return ss(A,B,C,D,dt)
end

#Y: the system impulse response 
#nx: model order
#nu: number of inputs
#dt: Y sample period
function ImpulseERA(Y,nx,nu,dt,p)
    ny,ns = size(Y);
    Y,ny,ns = CheckData(Y,ny,ns);
    return ERA_K(Y,nx,nu,dt,p,ny)
end

#bulid the U matrix presented in Dirceu's Thesis.
function BulidU(U,p,nu,ns) 
    U = U[:,1:ns-1];
    cut = ns-p  #cut parameter for lines
    Hu = zeros(nu*cut,ns-1);
    for i=1:cut
        Hu[(1:nu).+((i-1)*nu),i:ns-1] = U[:,1:ns-1-(i-1)];
    end
    return Hu
end

function GRA(Y,U,nx,dt,p)
    ny,ns = size(Y);
    nu, = size(U); #yes, for now, i'm putting some faith in user.
    Y,ny,ns = CheckData(Y,ny,ns);
    U,nu,ns = CheckData(U,nu,ns);

    Uh = BulidU(U,p,nu,ns);
    Yh = Y[:,1:ns-1];

    h = Yh*Uh'*inv(Uh*Uh');
    return ERA_K(h,nx,nu,dt,p,ny)
    

end


#------------------------------------------ARX RELATED FUNCTIONS-------------------------------------------------

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

    

    return array2mimo(G); #conversion to a mimo system
    #returns the space state form of the system, not the TF.
end


##-----------------------------PERCEPTRON F(U)--------------------------------------------

function TrainModel(model,Data)
    opt_state = Flux.setup(Adam(), model) 
    Flux.train!(model, Data, opt_state) do m, x, y
        Flux.mse(m(x), y)
    end
    return model
end

function SimpleData(Y,U,ns)
    U = [vec(U[:,i]) for i =1:ns] |> gpu
    H = [vec(Y[:,i]) for i =1:ns] |> gpu
    return zip(U ,H) |> gpu
end

function DataNARX(Y,U, ny, nu, na, nb,ns)
    H = zeros((nu*nb+(na+1)*ny),ns-na+1)
    Ur = [zeros(nu,nb-1) U]
    Yr = [zeros(ny,na-1) Y]
    
    for i=1:nb
        H[(1:nu).+((i-1)*nu),:] = Ur[:,(1:ns-na+1).+i]
    end

    for i=1:na+1
        H[(1:ny).+(nu*nb+(i-1)*ny),:] = Yr[:,(1:ns-na+1).+(i-1)]
    end
    
    Input  = [vec(H[1:(nu*nb+na*ny),i]) |> gpu for i=1:ns-na+1]
    Output = [vec(H[(1:ny).+(nu*nb+na*ny),i]) |> gpu for i=1:ns-na+1]

    return zip(Input, Output)
end

function NNARX_LSTM(dataSize,p,ny)
    model = Chain(
        Dense(dataSize,dataSize,tanh),
        LSTM(dataSize,dataSize), 
        LSTM(dataSize,1), 
        Dense(1,p,tanh),
        Dense(p,ny)
        ) |> gpu;
    return model
end

function DataNorm(Y)
    Ymax = maximum(abs.(Y),dims=2);
    Ymin = minimum(abs.(Y),dims=2);
    return (Y.-Ymin)./(Ymax-Ymin)
end

function Simple(V,type)
    nv, = size(V)
    F = [[type for i=1:nv-2]; identity] |> gpu
    neurons = [Dense(V[i],V[i+1],F[i]) for i=1:nv-1] |> gpu
    return Chain(neurons) |> gpu
end

function SimpleOpt(V,Data,type)
    model = TrainModel(
        Simple(V,type) |> gpu,
        Data
        )

    Yl = zeros(ny,ns) |> gpu
    j = 1
    for i in Data
            Yl[:,j] = model(i[1])
            j += 1
    end

    return Yl

end

function Testador(Data,Entradas,Saidas,largura,profundidade,passo)
    InfoGen = Matrix{Float64}(undef,largura,profundidade)
    for i = 1:passo:largura
        for j = 1:passo:profundidade
            V = Int.([Entradas; i*ones(j); Saidas])
            Yl = SimpleOpt(V,Data,relu)
            InfoGen[i,j] = norm((Y |> gpu) - Yl)/norm(Y)
        end
    end
    return InfoGen
end

#----------------------------------------------------------------NN AS IDENTIFIER--------------------------------------------------------------------


# function Testador(Data,Entradas,Saidas,largura,profundidade,passo)
#     InfoGen = Matrix{Float64}(undef,largura,profundidade)
#     for i = 1:passo:largura
#         for j = 1:passo:profundidade
#             V = Int.([Entradas; i*ones(j); Saidas])
#             model = TrainModel(Simple(V,relu),Data)
#             sysPest,sysDest = F2DOF(dt,[Float64(i) for i in ParamEst]);
#             Yest,_,_= lsim(sysDest,U',t,x0=x0); 
#             InfoGen[i,j] = norm((Y |> gpu) - Yest)/norm(Y)
#         end
#     end
#     return InfoGen
# end


function K(U,x0,model,na,nu,nb,ns)
    Hn = zeros(na*ny+nb*nu,ns)
    Ur = [zeros(nu,nb-1) U]
    for i=1:nb
        H[(1:nu).+((i-1)*nu),:] = Ur[:,(1:ns-na+1).+i]
    end

end