function [index_cov_samp, covar_id] = sample_pairwise(num_constraint, task_number,seed)
% num_constraint    % number of constraint

num_cov = task_number*(task_number-1)/2; % totoal number of covariance info.
 
 
num_pair =  num_cov*(num_cov-1)/2 ;

rng(seed);
pairs = randperm(num_pair);
pairs = pairs(1:num_constraint );
index_pair =zeros(num_pair,1); 
p = 0;
for ii = 1:num_cov-1
     index_pair(ii+p:ii+p+ num_cov-1-ii,1) = ii;
     p = p + num_cov -1- ii;
end

covar_id = zeros(num_constraint,2);          % a pair of selected covariance constraint
for ii = 1:num_constraint
    current_cov = index_pair(pairs(ii));
    [covar_id(ii,1), covar_id(ii,2)] = get_pair_index(current_cov,num_cov,pairs(ii));
end


p = 0;
index_cov =zeros(num_cov,2);
for ii = 1:task_number-1
     index_cov(ii+p:ii+p+ task_number-1-ii,1) = ii;
     index_cov(ii+p:ii+p+ task_number-1-ii,2) = ii;
     p = p + task_number -1- ii;
end


% rng(1024);
% pair = zeros(num_cov,2);          % a pair of covariance constraint
% pair(:,1) = randperm(num_cov);      
% pair(:,2) = randperm(num_cov);
% pair = pair(1:num_constraint,:);          % random select several constrain to encode it into
% 
% 
% % remove duplicate pairs
% index_removed = [];
% for ii = 1:num_constraint
%    if pair(ii,1) >= pair(ii,2)
%        index_removed = cat(1,index_removed,ii);
%    end
% end
% pair(index_removed,:) = [];

index_cov_samp =zeros(num_constraint,2);
index_cov_samp(:,1) = index_cov(covar_id(:,1),1);
index_cov_samp(:,2) = index_cov(covar_id(:,2),2);



end


function [i,j] = get_pair_index(current_task,task_num,pair_id)

        i = current_task;
        j = task_num - (task_num*i - i*(i+1)/2 - pair_id);
        
end