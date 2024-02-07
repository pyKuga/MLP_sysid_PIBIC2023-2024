
#tirar os zeros do inicio e refazer o proporcional r
function PulseGen(n)
    r = round(Int,n/4);
    Z = zeros(r);
    O = ones(r);
    S = [O; -O; Z;Z; 0];
    return S'
end

#gera o dotE de acordo com a formulação detalhada da IC
function dotEgen(X,u)

  x = X[1];
  θ = X[2];
  v = X[3];
  α = X[4];

  m2glsin = m2*l*sin(θ);
  m2glcos = m2*l*cos(θ);

  M = [
    m1+m2 -m2glcos;
    -m2glcos J0+m2*l^2 ;
  ];

  invM = inv(M);

  F = [
    c1*v+m2glsin*α^2 + k*x-u;
    beta*α + m2glsin
  ];

  
  return [
    v;
    α;
    -invM*F
    ];

end

function chirp(w,t)
  return sin.(w*t.^2)
end

#runge kutta 4
function RK4(X0,U,dt,maxT,n)
    T = 0:dt:maxT;
    X = zeros(size(X0)[1],n);
    X[:,1] = X0;
    h = dt;
    for i=1:n-1
      u = U[i];
      k1 = dotEgen(X[:,i],u);
      k2 = dotEgen(X[:,i]+0.5*k1*h,u);
      k3 = dotEgen(X[:,i]+0.5*k2*h,u);
      k4 = dotEgen(X[:,i]+0.5*k3*h,u);
      hPhi = h*(k1+2*k2+2*k3+k4)/6;
      X[:,i+1] = X[:,i] + hPhi;    
    end

    C = [1 0 0 0; 0 1 0 0];

    Y = C*X

    return Y,X,T
end

function metrica(Y1,Y2)
  return norm(Y2-Y1)/norm(Y1)
end