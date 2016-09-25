function realdataname = gen_mtl_fullrank_partialcorrupted_data_main(dataname,task_number, sample_size, feature_dim, sample_energe,...
                                 noise_level,noise_partial,ratio,dataDir)
% this script generate data
% addpath(genpath('./'));
% dataDir = '../../../../AllData/kMTRL/syn_MTRL/';
% timeflag = strrep(datestr(datetime),' ','');
% disp(timeflag);
% timeflag = strrep(timeflag,':','-');
% timeflag = timeflag(1:end-3);
% dataDir
 

%% set parameters
% task_number   = 10;
% sample_size   = 50;
% feature_dim   = 20;
% sample_energe = 10;
% noise_level   = 5; 
% rank_W        = 2;

% ratio = [0.2, 0.3]; %percentage of training dataset and testing dataset.



realdataname = strcat(dataDir, sprintf('%s_K%d_N%d_D%d_Nois%d.mat',dataname,task_number,sample_size,feature_dim,noise_level*100));


if exist(realdataname, 'file') ~= 2
disp('new data');
    
 

[data,label,W,W_rank_use,b] = mtl_syn_MTRL_bias_fullrank_partialcurrupted(task_number, sample_size, feature_dim,...
                                         sample_energe, noise_level,noise_partial);


[trainX,trainY,validX,validY,trainallX,trainallY, testX,testY] = split_data_zscoreXY(data,label,ratio);


save(realdataname);

end