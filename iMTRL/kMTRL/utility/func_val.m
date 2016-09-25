function  funcval = func_val(data, label, Omega, lambda1, lambda2, varargin)
  
  X = data;
  Y = label;
  W = varargin{1};

  K = size(W,2);
  numinput = nargin; 
  funcval = 0;
  Omega_inv = pinv(Omega);
  
   if numinput == 6
        
       for i = 1:K
             Xi = X{i};
             Yi = Y{i};
             ni = size(Xi,1);
             funcval = funcval + sum((Yi - Xi*W(:,i)).^2)/ni;
        end
        funcval = funcval + lambda1/2*trace(W'*W) + lambda2/2*trace(W*Omega_inv*W');
      
       
   else if numinput == 7
        b = varargin{2};
       for i = 1:K
             Xi = X{i};
             Yi = Y{i};
             bi = b(i);
             ni = size(Xi,1);
             funcval = funcval + sum((Yi - Xi*W(:,i)-bi).^2)/ni;
        end
        funcval = funcval + lambda1/2*trace(W'*W) + lambda2/2*trace(W*Omega_inv*W');
   
       
       end
       
  
 
 

end