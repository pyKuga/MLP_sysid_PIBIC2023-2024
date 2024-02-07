function F2DOF(dt,Param)
    m1,m2,c1,c2,c3,k1,k2,k3 = Param;

    M = [m1 0; 0 m2];
    C = [c1+c2 -c2; -c2 c2+c3];
    K = [k1+k2 -k2; -k2 k2+k3];

    Id = [1 0; 0 1];
    Z = [0 0; 0 0];

    invM = inv(M);

    A = [Z Id; -invM*K -invM*C];
    B = [Z; invM];
    B = B[:,1];
    C = [1 0 0 0; 0 1 0 0];
    #D = [0 0;0 0];

    sistema = ss(A,B,C,0);
    sistemaD = c2d(sistema,dt);

    return sistema, sistemaD 
end  
    
#gera um sinal normal normatizado manualmente
function noise_gen(seed,r,c)
    merst = MersenneTwister(seed); #mersenne twister generator
    signal = randn(merst, Float64,(r,c)); #gaussian noise
    mu = mean(signal,dims=2);
    signal_mu = signal.-mu;
    across = (maximum(signal.-mu,dims=2) - minimum(signal.-mu,dims=2))/2;
    return signal_mu./across
end


function GenParam(m_max,c_max,k_max,data_size)
    
    p = (0.01:0.01:1.01)
    M_it = p.*m_max
    K_it = p.*k_max
    C_it = p.*c_max

    m1 = rand(M_it)
    m2 = rand(M_it)
    k1 = rand(K_it,data_size)
    k2 = rand(K_it,data_size)
    k3 = rand(K_it,data_size)


    Param = [
        rand(M_it,data_size)
        rand(M_it,data_size)
        rand(C_it,data_size)
        rand(C_it,data_size)
        rand(C_it,data_size)
        rand(K_it,data_size)
        rand(K_it,data_size)
        rand(K_it,data_size)
        ]  

    Param = reshape(Param,(data_size,8))
    return [Param[i,:] for i=1:data_size]
end

function ParamGen(m_max,ξ_max,k_max)
    
    p = (0.01:0.0099:1)

    m1,m2 = rand(p,2)*m_max;
    k1,k2,k3 = rand(p,3)*k_max;
    
    ξ = rand(p,2)*ξ_max;

    M = [m1 0; 0 m2];
    K = [k1+k2 -k2; -k2 k2+k3];
    F = eigen(inv(M)*K);
    Ψ = F.vectors;
    ω = F.values;

    am = 2*sqrt.(ω);
    A = [ones(2) ω];

    v = inv(A)*am.*ξ;

    α = v[1];
    β = v[2];
    
    c2 = β*k2;
    c1 = α*m1+β*k1;
    c3 = α*m2+β*k3;

    return [m1,m2,c1,c2,c3,k1,k2,k3]
    
end

