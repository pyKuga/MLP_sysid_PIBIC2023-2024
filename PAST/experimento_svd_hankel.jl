using ToeplitzMatrices
using LinearAlgebra

A = [1 2 3 4; 5 6 7 8]
f = length(A)

esses = zeros(f,f)


for n in 2:f-1 
    H = copy(Hankel(A[1:n],A[n:f]));
    U,S,V = svd(H);
    esses[1:length(S),n] = S;
end
