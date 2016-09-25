function [model] = MTRL_Omegatrue_CVX_nobias(X, Y, init, d, K, lambda1, lambda2)

W = init.W;
 
Omega_inv = init.Omega_inv;

cvx_begin quiet
    variables W(d,K) 
    minimize( obj_func_nobias(X,Y, W, Omega_inv,d, K,lambda1,lambda2) )
cvx_end


model.W = W;

 
end