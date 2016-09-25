% optimize the objective: min_x 0.5 || Ax - y ||^2 + rho1 * ||x||_1
% A: n by d.
% y: n by 1.

function [W_new, b_new, funcVal] = kMTRL_fista_Wb(X,Y,model,parameters,Fista_options)

 
funcVal = [];

% flags of optimization algorithms .
bFlag = Fista_options.bFlag; % =0 this flag tests whether the gradient step only changes a little
tFlag = Fista_options.tFlag; % 3; % the termination criteria
maxIter = Fista_options.maxIter;
tol     = Fista_options.tol;

% initialize a starting point
W      = model.W;
b      = model.b; 

Omega_inv = model.Omega_inv;
lambda1 = parameters.lambda1; % parameter on complexity of W.
lambda2 = parameters.lambda2; % parameter on task relationship.
 


x0 = [W(:);b];
[d,K]  = size(W); % feature, number of tasks 

xk   = x0; % x_k
xk_1 = x0; % x_{k-1}

tk   = 1; % t_k
tk_1 = 0; % t_{k-1}

% initialize other variables.
L    = 1; % initial L       << here we use 0.1 so you can see how line search goes.
eta  = 1.1; % increment of L

iter = 0;
Lk = L;
while iter < maxIter
    alpha = (tk_1 - 1) /tk;

    % get current search point by a linear combination
    yk = (1 + alpha) * xk - alpha * xk_1;

    % compute function value and gradients of the search point
    fg_yk  = gradVal_eval (yk,X,Y,Omega_inv,d,K,lambda1,lambda2); % gradient
    fv_yk  = funVal_eval  (yk,X,Y,Omega_inv,d,K,lambda1,lambda2); % function value


    % start line search
    while true
%         next_point = yk-1/Lk*fg_yk;
        pL_yk = proximal(yk-1/Lk*fg_yk,lambda1/Lk);
         

        Fv_plyk = funVal_eval(pL_yk,X,Y,Omega_inv,d,K,lambda1,lambda2);
        delta   = pL_yk - yk;
        q_apro  = fv_yk + delta'* (fg_yk + Lk/2 * delta);


        % the line search procedure goes here!

        % compute the
        % no need to compute the non-smooth term.
        %
        if (Fv_plyk   <= q_apro)
            break;
        end
        Lk = Lk * eta;

    end


    % update current and previous solution.
    xk_1 = xk;
    xk = pL_yk;

    % concatenate function value (smooth part + non-smooth part).

    funcVal = cat(1, funcVal, Fv_plyk);

    if (bFlag)
        % fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end

    % test stop condition.
    switch(tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= tol)
                break;
            end
        case 3
            if iter>= maxIter
                break;
            end
    end

    % update other variables.
    iter = iter + 1;
    tk_1 = tk;
    tk = 0.5 * (1 + (1+ 4 * tk^2)^0.5);

end

x = pL_yk;

W_new = reshape(x(1:end-K), d, K);
b_new = x(end - K + 1: end);

% private functions


    % gradient of smooth part f at a given point x.
    function [grad_x] = gradVal_eval(x,X,Y,Omega_inv,d,K,lambda1,lambda2)
        % gradient goes here
         
         W = reshape(x(1:end-K),d,K);
         b = x(end-K+1:end);
         grad_W = zeros(d,K);
         grad_b = zeros(K,1);
         for i = 1:K
             Xi = X{i};
             Yi = Y{i};
 
             ni = size(Xi,1);
             grad_W(:,i) = grad_W(:,i) - 2*Xi'*(Yi - Xi*W(:,i)-b(i))/ni;
             
             grad_b(i) = grad_b(i) - 2* sum(Yi - Xi*W(:,i)-b(i))/ni;
         end
         
         grad_W = grad_W + W*(lambda1*eye(K)+lambda2*Omega_inv);    
         grad_x = [grad_W(:);grad_b];
         
    end

    % function value of smooth part f.
    function [funcVal] = funVal_eval (x,X,Y,Omega_inv,d,K,lambda1,lambda2)
        % function value goes here.
        funcVal = 0;
%         W = reshape(x,d,K);
        W = reshape(x(1:end-K),d,K);
        b = x(end-K+1:end);
        for i = 1:K
             Xi = X{i};
             Yi = Y{i};
             ni = size(Xi,1);
             funcVal = funcVal + sum((Yi - Xi*W(:,i)-b(i)).^2)/ni;
        end
         funcVal = funcVal + lambda1/2*trace(W'*W) + lambda2/2*trace(W*Omega_inv*W');
    end
 

end

% projection
function z = proximal (v, beta)
    % this projection calculates
    % argmin_z = 0.5 * \|z-v\|_2^2 + beta \|z\|_1
    % z: solution
    % l1_comp_val: value of l1 component (\|z\|_1)
    z = v;
end
