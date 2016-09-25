% this script generate data
addpath(genpath('../'));
dataDir = '../../../../AllData/kMTRL/syn/';
timeflag = strrep(datestr(datetime),' ','');
disp(timeflag);
timeflag = strrep(timeflag,':','-');
timeflag = timeflag(1:end-3);

dataname = strcat('mtl_lr_split',timeflag);

%% set parameters
task_number   = 10;
sample_size   = 100;
feature_dim   = 5;
sample_energe = 10;
noise_level   = 10; 
rank_W        = 4;

ratio = [0.2, 0.3]; %percentage of training dataset and testing dataset.


if rank_W >= feature_dim || rank_W >= task_number
    error('rank of W too large, cannot larger than feature dimension or task number');
end


[data, label,W] = mtl_syn_lw(task_number, sample_size, feature_dim, sample_energe, noise_level,rank_W);

% [data, label,W,W_rank] = mtl_syn_w_cov(task_number, sample_size, feature_dim, sample_energe, noise_level);
[trainX,trainY,validX,validY,trainallX,trainallY, testX,testY] = split_data(data,label,ratio);



save(strcat(dataDir, sprintf('%s_K%d_N%d_D%d_nois%d',dataname,task_number,sample_size,feature_dim,noise_level)), 'trainX','trainY','validX','validY',...
    'testX','testY','trainallX', 'trainallY','W','data', 'label',...
    'sample_energe', 'ratio', 'noise_level');