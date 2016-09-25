function [rmse, rmse_all] = make_evaluation(Xtest,Ytest, varargin)
% input is either cell or matrix of test data, Xtest,Ytest should be both
% cell or matrix.
 numinput = nargin; 
 
  W = varargin{1};
 K = size(W,2);
 rmse_all = zeros(K,1);
 
 
 if numinput ==3
  
 for i = 1:K
     Xi=Xtest{i};
     Wi = W(:,i);   
 
 
      Y_predi = Xi*Wi;
 
 
     rmse_all(i) = sqrt(mean((Y_predi - Ytest{i}).^2));
 end
 rmse = mean(rmse_all); % average rmse for all tasks.
     
 else if numinput ==4
     b = varargin{2};
    for i = 1:K
         Xi=Xtest{i};
         Wi = W(:,i);   
         bi = b(i);
          
 
         Y_predi = Xi*Wi + bi; 
         rmse_all(i) = sqrt(mean((Y_predi - Ytest{i}).^2));
    end
    rmse = mean(rmse_all); % average rmse for all tasks.
     
         
 end





end