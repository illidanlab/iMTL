function model =  kMTRL_kernel_activelearning_interface(data, label,pair, Omega_true,W_true,testX,testY, Options)
% this script tune the parameters of kMTRL on validation set and return the best model


 
tunedParas = Options.tunedParas;

% parallel computing parameters  %%%%%%%%%%%%%%%%%
lenPara = 1;
paraNames = fieldnames(tunedParas);
num_paras = length(paraNames);
len_paras = zeros(num_paras,1);
dividend  = ones(num_paras,1);   

para_i = cell(num_paras,1);
for i = 1:num_paras
    para_i{i} = getfield(tunedParas, paraNames{i});
    lenPara = lenPara*length(para_i{i});
    len_paras(i) = length(para_i{i});  % lenght of each parameter array
    
    if i ~= num_paras
        for jj = i+1:num_paras
            dividend(i) = dividend(i) * length(getfield(tunedParas, paraNames{jj}));
        end
    else
        dividend(i) = len_paras(i);
    end
end

%% Initialization
K = length(label);
d = size(data{1},2);
init.Omega_true = Omega_true;
init.c    = Options.c;
init.maxIter = Options.maxIter;
init.pair = pair;

label_new = cell(K,1);
for i = 1:K
    yy = label{i};
    label_new{i} = yy';  
end

models = cell(lenPara,1);
RMSE   = zeros(lenPara,1);
funcvals   = zeros(lenPara,1);
train_RMSE   = zeros(lenPara,1);
normall   = zeros(lenPara,1);
%% Grid search
parfor  i = 1:lenPara

        paras_indexs = zeros(num_paras,1); % the real index in each array
        for j = 1:num_paras
            if j == num_paras
                paras_indexs(j) = mod(i,len_paras(j));

            else
                temp = ceil(i/dividend(j));
                paras_indexs(j) = mod(temp,len_paras(j));
            end 
            if paras_indexs(j)== 0
                    paras_indexs(j) = len_paras(j);
            end
        end


        parameters = struct();
        for jj = 1:num_paras
            para_array = para_i{jj}; % current parameters arrays.
            parameters.(paraNames{jj}) = para_array(paras_indexs(jj)) ;
        end

        lambda1 = parameters.(paraNames{1});
        lambda2 = parameters.(paraNames{2});
        
        model = kMTRL_kernel_activelearning_explore(init, data,label_new,'linear',0,lambda1,lambda2);
        
        models{i} = model;
        [RMSE(i),b]   = make_evaluation(testX,testY, model.W, model.b);
        [train_RMSE(i),b]   = make_evaluation(data,label, model.W, model.b);
        funcvals(i) = model.func_all(end);
        normall(i) = norm(model.W-W_true,2);
end
[~, index] = min(RMSE);
% [~, index] = min(normall);
model = models{index};
model.index   = index;
model.RMSE    = RMSE;
model.normall = normall;
model.train_RMSE = train_RMSE;
model.funcvals = funcvals;
end