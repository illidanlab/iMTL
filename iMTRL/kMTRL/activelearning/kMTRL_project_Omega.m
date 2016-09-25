function [Omega , output] = kMTRL_project_Omega(W,Omega_truth, pair,c)
% use the part of ground truth pairwise information to 
 
    maxIter = 100;
    tol     = 1e-9;
%   epsilon = 1e-3;    
    task_num = size(W,2);
    A = W'*W;
    [V, D] = eig(A);
 
    
    A2 = V*(abs(D).^(1/2))/V;
    Omega = A2/trace(A2);

    flag = 0;
    if ~isempty(pair)  
    iter = 0;
    Omega_new = Omega;
    num_constraint = size(pair,2);
    
    
    while iter < maxIter
        
        flag_constraint = 0;
        for ii = 1:num_constraint
            % get the pair of current constraint 
            i1  = pair(1,ii); 
            j1  = pair(2,ii); 
            i2  = pair(3,ii); 
            j2  = pair(4,ii); 
             
            % projection
            if (Omega(i1,j1)-Omega(i2,j2))*(Omega_truth(i1,j1)-Omega_truth(i2,j2)) < 0  
                if (Omega_truth(i1,j1)-Omega_truth(i2,j2)) > 0
                    Omega_pair1 = (c + Omega(i1,j1) + Omega(i2,j2))/2;            % Omega(i1,j1) > Omega(i2,j2) 
                    Omega_pair2 = (-c + Omega(i1,j1) + Omega(i2,j2))/2;
                else
                    Omega_pair2 = (c + Omega(i1,j1) + Omega(i2,j2))/2;          % Omega(i1,j1) < Omega(i2,j2) 
                    Omega_pair1 = (-c + Omega(i1,j1) + Omega(i2,j2))/2;
                end
                Omega(i1,j1) = Omega_pair1;
                Omega(i2,j2) = Omega_pair2;
                Omega(j1,i1) = Omega(i1,j1);
                Omega(j2,i2) = Omega(i2,j2);
                 flag = 1;
                 flag_constraint = 1;
                  
            end
             
        end
        iter = iter + 1;
        
        c = 0.9*c; 
        [V, D] = eig(Omega);   
        D(D<0) = 0 ; 
        A2 = V*D/V; 
        Omega = A2/trace(A2);      
        
        Omega_new = real(Omega);
        
        
        if norm(Omega_new - Omega,'fro') < tol && flag_constraint == 0
            break;
        end
        
        
    end     

    Omega = Omega_new ;
    end        
    
     
    output.flag = flag;
 
end

 