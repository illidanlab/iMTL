function model = kMTRL_active_learning_Wb(data, label, init, Omega_truth, pair, Options,parameters)

X =  data;
Y =  label;

K = length(label);
d = size(X{1},2);

maxIter = Options.maxIter;
tol     = Options.tol;
 
Fista_options = Options.Fista_options;
 
 
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
    
    funcvals = func_val(X, Y, Omega, parameters.lambda1, parameters.lambda2, W, b);
    
    % Compute W given Omega.
    [W,b,funcVal]= kMTRL_fista_Wb(X,Y,model,parameters,Fista_options);
%      plot(funcVal);
    W = real(W);
    % Compute Omega given W.
    [Omega,output] = kMTRL_project_Omega(W, Omega_truth, pair, c);
    Omega = real(Omega);
    
    if (output.flag == 1)
        flag = 1;
    end
    
    if sum(sum((real(model.W) - real(W)).^2)) < tol*K*d && sum(sum((real(model.Omega) - real(Omega)).^2)) 
        break;
    end
 
    model.Omega = Omega;
    model.Omega_inv = pinv(Omega);
    model.W     = W;
    model.b     = b;
     
    func_all = cat(1, func_all,funcvals);
    
  
    iter = iter + 1;
    
end
func_all = cat(1, func_all,funcvals);
 
model.func_all = func_all;
% profile viewer;
%     if( flag == 1)
%         disp('projected')
%     end
% iters
end