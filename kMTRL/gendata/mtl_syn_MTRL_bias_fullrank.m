% Generate multi-task dataset.
% this script construct a W that some columns has 1 or -1 correlation with
% orthers
function [data, label,W,W_rank, b] = mtl_syn_MTRL_bias_fullrank(task_number, sample_size, feature_dim, sample_energe, noise_level)
% noise_level = 5;
% sample_energe = 10;
% feature_dim = 5;
% task_number = 10;
% rank_W = 4;
 
rng(1024);
W = rand(feature_dim,task_number);
% W = randi([-10,10],feature_dim,task_number);
% WT = gen_lowrank_matrix(rank_W,task_number,feature_dim,1024) ;
% W = WT';
% for i = 1:floor(task_number/2.3)
%    W(:,i) = (-1)^i*W(:,task_number+1 - i); 
% end
% [U,S,V] = svd(W);

W_rank = rank(W);
 
      
%% Generate Data
 
 
% this feature dimension includes the constant b

data  = cell(1,task_number);
label = cell(1,task_number);

rng(1024)
b = rand(task_number,1);

num = sample_size;
for i = 1:task_number
   rng(i)
   X = randn(num,feature_dim )*sample_energe;
   
   rng(100*i)
   y = randn(num,1)*noise_level;

   w = W(:,i);
   y = y + X*w + b(i);

   label{i} = y;
   data{i}  = X;

end




end