function [pair] = query_strategy(W, Omega, num_constraints)
% return the task id of pairs in the following form:  
      pair = [];
      if num_constraints ~= 0 

      pair = zeros(4,num_constraints); % pair= [i1,j1,i2,j2;....]

      A = W'*W;
      [V, D] = eig(A);
      A2 = V*(abs(D).^(1/2))/V;
      Omega_true = A2/trace(A2);
      
%       Omega_true = (W'*W)^(1/2);
%       Omega_true = Omega_true/trace(Omega_true);
%       Omega_true = Omega_true/trace(Omega_true);
      
      task_number = size(W,2);
      
      num_cov = task_number*(task_number-1)/2; % totoal number of covariance info.
      num_pair =  num_cov*(num_cov-1)/2 ;
      
      all_pairs_true = zeros(num_pair,1);  % all pairs of distance of true Omega.
      all_pairs_learn = zeros(num_pair,1);  
      
      omega_true_vec = get_upper_vec(Omega_true);
      omega_learn_vec =  get_upper_vec(Omega);
      
      p = 1;
      for i = 1:num_cov-1
          cnp = num_cov - i;%number of pairs for current covariance
          all_pairs_true(p:p+cnp-1) = omega_true_vec(i) - omega_true_vec(i+1:end);
          all_pairs_learn(p:p+cnp-1) = omega_learn_vec(i) - omega_learn_vec(i+1:end);
          p = p + cnp;
      end
      
      sign_pairs = sign(all_pairs_true.*all_pairs_learn); % negative means order is not consistent
      
      dis_pair  = sign_pairs.*abs(all_pairs_true - all_pairs_learn);
      
%       store_n = zeros(num_constraints,1);
      for ii = 1:num_constraints
          if length(find(dis_pair<0)) ~= 0
              [m,n] = min(dis_pair);
          else
              [m,n] = max(dis_pair);
          end

%           [m,n] = min(dis_pair); % value of distance, n index of most disorder covariance pair;
          % decoce n to 2 covariance id and then to 4 tasks numbers:
          [cov_row, cov_column] = decode_triangular_id(n,num_cov);

          [pair(1,ii),pair(2,ii)] = decode_triangular_id(cov_row,task_number);

          [pair(3,ii),pair(4,ii)] = decode_triangular_id(cov_column,task_number);

          dis_pair(n) = 0;
%           store_n(ii) = n;
      end
      
      end
end

 


 