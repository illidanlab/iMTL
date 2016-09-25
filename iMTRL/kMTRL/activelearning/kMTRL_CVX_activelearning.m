function model = kMTRL_CVX_activelearning(data, label, init, Omega_truth, pair, Options,parameters)

X =  data;
Y =  label;

K = length(label);
d = size(X{1},2);

maxIter = Options.maxIter;
tol     = Options.tol;
 
 
 
 
c = Options.c;
% initialization 
iter = 0;
 
Omega       = init.Omega;
model.Omega = Omega;
model.Omega_inv = init.Omega_inv;
W           = init.W;
model.W = W;
b           = init.b;
model.b = b;
 
    
flag = 0;
% profile on;
func_all = [];
while iter < maxIter
    
    funcvals = func_val(X, Y, Omega, parameters.lambda1, parameters.lambda2, model.W, model.b);
    
    % Compute W given Omega.
    [model_cvx] = MTRL_Omegatrue_CVX(X, Y, model,...
                    d, K, parameters.lambda1, parameters.lambda2);

    
    % Compute Omega given W.
    [Omega,output] = kMTRL_project_Omega(model_cvx.W, Omega_truth, pair, c);
    if (output.flag == 1)
        flag = 1;
    end
    
    if sum(sum((model.W - model_cvx.W).^2)) < tol*K*d  
        break;
    end
 
    
    model.W     = model_cvx.W;
    model.b     = model_cvx.b;
    model.Omega = Omega;
    model.Omega_inv = pinv(Omega);
 
     
    func_all = cat(1, func_all,funcvals);
    
  
    iter = iter + 1;
    
end
func_all = cat(1, func_all,funcvals);
% iter
model.func_all = func_all;
% profile viewer;
%     if( flag == 1)
%         disp('projected')
%     end
% iters
end