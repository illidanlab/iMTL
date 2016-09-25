function realdataname = gen_mtl_data_main(dataname,task_number, sample_size, feature_dim, sample_energe, noise_level, rank_W,ratio,dataDir)
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



realdataname = strcat(dataDir, sprintf('%s_K%d_N%d_D%d_Nois%d_rankW%d.mat',dataname,task_number,sample_size,feature_dim,noise_level*100,rank_W));


if exist(realdataname, 'file') ~= 2
disp('new data');
    
if rank_W >= feature_dim || rank_W >= task_number
    error('rank of W too large, cannot larger than feature dimension or task number');
end


[data,label,W,W_rank_use,b] = mtl_syn_MTRL_bias(task_number, sample_size, feature_dim,...
                                         sample_energe, noise_level, rank_W);


[trainX,trainY,validX,validY,trainallX,trainallY, testX,testY] = split_data(data,label,ratio);


save(realdataname);

end