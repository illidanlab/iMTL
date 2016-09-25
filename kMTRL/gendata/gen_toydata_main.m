function realdataname = gen_toydata_main(dataname, sample_size, sample_energe, noise_level,ratio,dataDir)

 
 
    WT = [[3];      % the last element in each column is b.
          [-3];
          [0];
%           [5, 12];
%           [-5, -15]
          ];
%  rng(1024);
%     WT = randi([-10,10], 3,10);
 
b = [10; -5; 1];

 
%% Generate Data
W = WT';

[feature_dim, task_number] = size(W);

realdataname = strcat(dataDir, sprintf('%s_K%d_N%d_D%d_Nois%d.mat'...
                ,dataname,task_number,sample_size,feature_dim,noise_level*100));    




data  = cell(1,task_number);
label = cell(1,task_number);



for i = 1:task_number
%    rng(1024)
   X = rand(sample_size,feature_dim)*sample_energe;


%    rng(1024)
   y = rand(sample_size,1)*noise_level;

   w = W(:,i);
   y = y + X*w + b(i);

   label{i} = y;
   data{i}  = X;

end
    
[trainX,trainY,validX,validY,trainallX,trainallY, testX,testY] = split_data(data,label,ratio);

save(realdataname);

end