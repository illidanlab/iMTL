function [pair] = query_strategy_random(task_number,num_constraints,seed)
% return the task id of pairs in the following form:       
 
      pair = zeros(4,num_constraints); % pair= [i1,j1,i2,j2;....]

 
      
      num_cov = task_number*(task_number-1)/2; % totoal number of covariance info.
      num_pair =  num_cov*(num_cov-1)/2 ;
      
      rng(seed);%1024
      pair_index = randperm(num_pair); 
       
%       store_n = zeros(num_constraints,1);
      for ii = 1:num_constraints
          n = pair_index(ii); % index of current randomly choose 
          % decoce n to 2 covariance id and then to 4 tasks numbers:
          [cov_row, cov_column] = decode_triangular_id(n,num_cov);

          [pair(1,ii),pair(2,ii)] = decode_triangular_id(cov_row,task_number);

          [pair(3,ii),pair(4,ii)] = decode_triangular_id( cov_column,task_number);
          
      end
end

 


 