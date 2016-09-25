%% FUNCTION gen_lowrank_matrix
% this script generate low rank weight matrix for multi task. 

%% INPUT 
% rank - the rank of matrix. 
% M, N - the size of matrix. 

%% OUTPUT
% rank matrix with the size of M*N. 

%% Codes starts here. 
function W = gen_lowrank_matrix(rank,M,N,seed)

rng(seed)
W = rand(M, N);
[U,S,V] = svd(W);
S = S(1:rank,1:rank);
W = U(:,1:rank)*S*V(:,1:rank)';


end