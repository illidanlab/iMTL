function [model] = MTRL_Omegatrue_CVX(X, Y, init, d, K, lambda1, lambda2)

W = init.W;
b = init.b;
Omega_inv = init.Omega_inv;

cvx_begin quiet
    variables W(d,K) b(K,1) 
    minimize( obj_func(X,Y, W,b, Omega_inv,d, K,lambda1,lambda2) )
cvx_end


model.W = W;
model.b = b;
 
end