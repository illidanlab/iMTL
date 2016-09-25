%%%%%%%% This scirpt generate the syn data to explore active learning and
%%%%%%%% randomly sample constraints
%%%%%%%% To vary the number of constraints and perform 5 cross validation 
%%%%%%%% To explore the significant of pair wise information with the
%%%%%%%% magnitude of Omega.


clear;
addpath(genpath('./'));
%%%%%%% This compares encoding true Omega and kMTRL
%% Load data Training, validation, Testing
timeflag = strrep(datestr(datetime),' ','');
disp(timeflag);
timeflag = strrep(timeflag,':','-');
timeflag = timeflag(1:end-3);
 
dataDir = '../syn_data/';
dataDir_res = '../results/';
 
dataname = strcat('mtl_bias_zscored_partialcorrupt');
task_number   = 10;
sample_size   = 25;
sample_energe = 10;
noise_level   = 10; 
ratio = [0.2, 0.3];
feature_dim   = 20;

noise_partial = [3,3,3,1,1,1,1,1,1,1];


realdataname = gen_mtl_fullrank_partialcorrupted_data_main(dataname,task_number, sample_size, feature_dim, sample_energe,...
                                 noise_level,noise_partial,ratio,dataDir);

 
data_name_generated = sprintf('%s_K%d_N%d_D%d_Nois%d.mat',dataname,task_number,sample_size,feature_dim,noise_level*100);
disp(realdataname);                             
data_load = load(realdataname);
trainallX = data_load.trainallX;
trainallY = data_load.trainallY;
testX = data_load.testX;
testY = data_load.testY;

flag = 0; % active : 1, random: 0

          
num_interacts = ones(1,8)*5;  
len_interact  = length(num_interacts);
num_fold = 5;              % number of cross validation fold.  
rmse_all = zeros(num_fold,len_interact+1);
norm_all = zeros(num_fold,len_interact+1);
norm_Omega = zeros(num_fold,len_interact+1);
K = task_number;

configureFile;


    
 
W_real = data_load.W;
Omega_real = (W_real'*W_real)^(1/2);
Omega_real = Omega_real/trace(Omega_real);

models_saved = cell(num_fold, len_interact+1);
 
for cross_num = 1:num_fold % cross validation
    
        trainX = cell(1,K);
        trainY =  cell(1,K);

        validX =  cell(1,K);
        validY =  cell(1,K);

        for kk = 1:K
            Xt = trainallX{kk};
            Yt = trainallY{kk};

            [num, feature] = size(Xt);
            
            rng(cross_num);
            index = randperm(num);

            train_num = round(num*ratio(1)/(ratio(1)+ratio(2)));
            valid_num = num - train_num;

            train_index = index(1:train_num);
            valid_index = index(train_num+1:end);

            trainX{kk} = Xt(train_index,:);
            trainY{kk} = Yt(train_index,:);

            validX{kk} = Xt(valid_index,:);
            validY{kk} = Yt(valid_index,:);

        end
        
        
        pair_act = [];
        pair_rand = [];
        for ii = 1:len_interact+1
 
        if flag == 1

        %%  kMTRL kernel active learning.
            modelKerAct = kMTRL_kernel_activelearning_interface(trainX,trainY,pair_act,Omega_real,W_real,validX,validY, ...
                                     configurePara.kMTRL_kernel_activelearning);
             method = 'Query Strategy';
 
             
            [rmse_all_temp, b]  = make_evaluation(testX, testY, modelKerAct.W, modelKerAct.b);
            norm_all_temp       =  norm(modelKerAct.W-W_real, 'fro');
            rmse_all(cross_num,ii) = real(rmse_all_temp);
            norm_all(cross_num,ii) = real(norm_all_temp);
            norm_Omega(cross_num,ii) = norm(modelKerAct.Omega - Omega_real,'fro');
 
            modelKerAct.pair_act = pair_act;            
            models_saved{cross_num,ii} = modelKerAct;
 
            if ii ~=len_interact+1        
               Omega_all = modelKerAct.Omega_all;
               pair_act_curr = query_strategy(real(modelKerAct.W), modelKerAct.Omega, num_interacts(ii));
               pair_act = [pair_act, pair_act_curr];
            end
        else
         
            %% kMTRL kernel random sample pairwise info
            modelKerRand = kMTRL_kernel_activelearning_interface(trainX, trainY,pair_rand,Omega_real,W_real,validX,validY,...
                                               configurePara.kMTRL_kernel_random);
 
             
            method = 'Random Selection';
            norm_all_temp1  = norm(modelKerRand.W-W_real, 'fro');
            [rmse_all_temp1,b]  = make_evaluation(testX, testY, modelKerRand.W,modelKerRand.b);

            rmse_all(cross_num,ii) = real(rmse_all_temp1);
            norm_all(cross_num,ii) = real(norm_all_temp1);
            norm_Omega(cross_num,ii) = norm(modelKerRand.Omega - Omega_real,'fro');
            modelKerRand.pair_rand = pair_rand;
            models_saved{cross_num,ii} = modelKerRand;

            if ii ~=len_interact+1
             pair_rand_curr = query_strategy_random(K,num_interacts(ii),ii);
             pair_rand = [pair_rand, pair_rand_curr];
            end
        end
        
 

        end
        
        
end

 
num_round = len_interact+1;
num_methods = 1;
rmse_mean = zeros(num_methods,num_round);
norm_mean = zeros(num_methods,num_round);
norm_Omega_mean = zeros(num_methods,num_round);

rmse_std = zeros(num_methods,num_round);
norm_std = zeros(num_methods,num_round);

for i = 1:num_round
 
        a = rmse_all(:,i);
        b = norm_all(:,i);
        rmse_mean(i) = mean(a);
        norm_mean(i) = mean(b);
        rmse_std(i) = std(a);
        norm_std(i) = std(b);
        cc = norm_Omega(:,i);
        norm_Omega_mean(i) = mean(cc);        
 
end    
 
disp(method)
disp(rmse_mean)
save(strcat(dataDir_res,data_name_generated,'_ActvsRand_results',timeflag,'.mat')); 
 
 