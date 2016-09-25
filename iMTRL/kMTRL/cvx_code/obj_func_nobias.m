function y  = obj_func_nobias(X,Y, W,  Omega_inv, d, K,lambda1,lambda2)
%addpath('../alg2_tensor_prox/')

        y = 0;
         
        for i = 1:K
             Xi = X{i};
             Yi = Y{i};
             ni = size(Xi,1);
             y = y + sum((Yi - Xi*W(:,i)).^2)/ni;
        end
         
        for i = 1:d
            wi = W(i,:);
            y = y +  lambda2/2*quad_form(wi',Omega_inv);
        end
%         
        y = y + lambda1/2*sum(sum(W.^2));% + lambda2/2*quad_form(W',Omega_inv);
        
end