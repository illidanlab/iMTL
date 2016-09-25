%%%%%%%% This scirpt generate the syn data to explore encoding true Omega
%%%%%%%% to MTRL and MTRL which is better. 
%%%%%%%% To vary the number of features and perform 5 cross validation 


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
 
dataname = strcat('mtl_bias');
task_number   = 10;
sample_size   = 25;
sample_energe = 10;
noise_level   = 5; 
ratio = [0.2, 0.3];
feature_dims   = 20:10:200;

number_feats = length(feature_dims);
num_fold = 5;              % number of cross validation fold.
rmse_all = zeros(2,number_feats,num_fold);
norm_all = zeros(2,number_feats,num_fold);
K = task_number;
for ii = 1:number_feats

    feature_dim = feature_dims(ii);
    realdataname = gen_mtl_fullrank_data_main(dataname,task_number, sample_size, feature_dim, sample_energe,...
                                     noise_level,ratio,dataDir);

    data_load = load(realdataname);

    trainallX = data_load.trainallX;
    trainallY = data_load.trainallY;
    testX = data_load.testX;
    testY = data_load.testY;
    
    
    for cross_num = 1:num_fold
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

 


        configureFile;

        W_real = data_load.W;
        Omega_real = (W_real'*W_real)^(1/2);
        Omega_real = Omega_real/trace(Omega_real);

        init = [];


        %%  MTRL 
        model_MTRL = MTRL_interface(init,trainX, trainY, validX, validY, W_real, configurePara.MTRL);
        [rmse_all_temp, b]   = make_evaluation(testX, testY, model_MTRL.W, model_MTRL.b);
%         norm_all_temp       = norm(model_MTRL.W-W_real, 2);
        norm_all_temp       = norm(model_MTRL.W-W_real, 'fro');
        rmse_all(1,ii,cross_num) = rmse_all_temp;
        norm_all(1,ii,cross_num) = norm_all_temp;


        %%  MTRL encode true Omega KERNEL
        modelmtrl = MTRL_Omegatrue_interface(trainX, trainY,Omega_real,W_real,validX,validY, configurePara.MTRL_Omegatrue);

        norm_all_temp1       = norm(modelmtrl.W-W_real, 'fro');
        [rmse_all_temp1,b]  = make_evaluation(testX, testY, modelmtrl.W,modelmtrl.b);

        rmse_all(2,ii,cross_num) = rmse_all_temp1;
        norm_all(2,ii,cross_num) = norm_all_temp1;

    end


end

rmse_mean = zeros(2,number_feats);
norm_mean = zeros(2,number_feats);


rmse_std = zeros(2,number_feats);
norm_std = zeros(2,number_feats);

for i = 1:number_feats
    for j = 1:2
        a = rmse_all(j,i,:);
        b = norm_all(j,i,:);
        rmse_mean(j,i) = mean(a);
        norm_mean(j,i) = mean(b);
        rmse_std(j,i) = std(a);
        norm_std(j,i) = std(b);
    end
end    

results_name = strcat(strrep(realdataname,'.mat',''),'results',timeflag);
save(results_name);

eMTRL_plot;
