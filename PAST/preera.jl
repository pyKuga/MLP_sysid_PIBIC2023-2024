using LinearAlgebra
using ToeplitzMatrices

A = [1,2,3,4,5]
B = [5,0,0,0,0]

H = Hankel(A,B)

H1 = H[:,1:size(H)[2]-1]; #puxa a primeira parte dela
H2 = H[:,2:size(H)[2]]; #puxa a segunda parte dela

U, S, V = svd(H1); #faz a SVD

G = diagm(S).^0.5
Q = inv(G)

A = Q*U'*H2*V*Q
C = U*G();
B = G*V';