% Generate multi-task dataset.

function [data, label, W] = mtl_syn_lw(task_number, sample_size, feature_dim, sample_energe, noise_level,rank_W)
% noise_level = 5;
% sample_energe = 10;
% feature_dim = 5;
% task_number = 10;
% rank_W = 4;
WT = gen_lowrank_matrix(rank_W,task_number,feature_dim,1024) ;
      
%% Generate Data
W = WT';
 

 
 % this feature dimension includes the constant b

data  = cell(1,task_number);
label = cell(1,task_number);


num = sample_size;
for i = 1:task_number
   rng(1024)
   X = rand(num,feature_dim )*sample_energe;
   
%    rng(3)
   y = rand(num,1)*noise_level;

   w = W(:,i);
   y = y + X*w;

   label{i} = y;
   data{i}  = X;

end




end