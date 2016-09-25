% this script generate a specific constructed data
addpath(genpath('../'));
dataDir = '../../../../AllData/kMTRL/syn/';
timeflag = strrep(datestr(datetime),' ','');
disp(timeflag);
timeflag = strrep(timeflag,':','-');
timeflag = timeflag(1:end-3);

dataname = strcat('mtl_specific_split',timeflag);



num = 20;
num_valid = 20;
num_test  = 20;
sample_size = num + num_valid + num_test;
noise_level = 5;
sample_energe = 10;
feature_dim = 5;
task_number = 10;
rank_W = 4;
WT = gen_lowrank_matrix(rank_W,task_number,feature_dim,1024) ;
      
%% Generate Data
W = WT';
% disp('ground truth');
% disp(W);

[feature_dim, task_number] = size(W);
 % this feature dimension includes the constant b

trainX  = cell(1,task_number);
trainY = cell(1,task_number);
validX =  cell(1,task_number);
validY =  cell(1,task_number);

testX  =  cell(1,task_number);
testY  =  cell(1,task_number);

trainallX = cell(1,task_number);
trainallY = cell(1,task_number);


for i = 1:task_number
   rng(1024)
   X = rand(num_valid+num_test,feature_dim )*sample_energe;
%    X = [X,ones(num,1)];
   y = rand(num_valid+num_test,1)*noise_level;

   w = W(:,i);
   y = y + X*w;

   rng(1024)
   X1 = rand(num,feature_dim )*sample_energe;
   y1 = rand(num,1)*noise_level;
   trainY{i} = y1 + X1*w;
   trainX{i}  = X1;
   
%     trainallX{i} = X(1:num+num_valid,:);
%     trainallY{i} = y(1:num+num_valid,1);
%     testX{i}     = X(num+num_valid+1:end,:);
%     testY{i}     = y(num+num_valid+1:end,1);
    validX{i} = X(1:num_valid,:);
    validY{i} = y(1:num_valid,1);
    trainallX{i} = [X1;X(1:num_valid,:)];
    trainallY{i} = [y1;y(1:num_valid,1);];
 
   testX{i}      =  X(num_valid+1:end,:);
    testY{i}     = y(num_valid+1:end,1);

end
save(strcat(dataDir, sprintf('%s_K%d_N%d_D%d_nois%d',dataname,task_number,sample_size,feature_dim,noise_level)));

