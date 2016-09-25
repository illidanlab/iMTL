function model = kMTRL_Omega_fixed_Wb(data, label,init, Options,parameters)

X =  data;
Y =  label;
 
K = length(label);
d = size(X{1},2);

Fista_options = Options.Fista_options;
% parameters = Options.parameters;
 
model.Omega_inv = init.Omega_inv;
model.Omega = init.Omega_true;
W           = init.W;
model.W = W;
model.b = init.b;
tol  = Options.tol;

iter = 0;
% Compute W given Omega.
func_val = [];

% while iter < Options.maxIter
% Compute W given Omega.
[W,b, funcVal] = kMTRL_fista_Wb(X,Y,model,parameters,Fista_options);

%plot(funcVal);
model.W     = W; 
model.b     = b; 
% iter = iter + 1;

%     if sum(sum((model.W - W).^2)) < tol
%         break;
%     end
% func_val = cat(1, func_val, funcVal(end));
 

% end
% iter
model.func_val = funcVal;
end