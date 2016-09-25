function output = kMTRL_FISTA_activelearning_Wb_interface...
                  (trainX, trainY, validX, validY,init, Omega_truth, pair, Options)
              
              

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


K = length(trainY);
d = size(trainX{1},2);
if isempty(init)
    Omega = eye(K);
    init.Omega = Omega/K;
    init.Omega_inv = pinv(init.Omega);
    rng(1024)         % this random seed cannot be same as the one generate data.
    W = rand(d,K);
    init.W = W;
    rng(1024)
    init.b = rand(K,1);
end
 

models = cell(lenPara,1);
RMSE   = zeros(lenPara,1);
RMSE_train = zeros(lenPara,1);
final_func = zeros(lenPara,1);
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
        
        
        parameters.lambda1 = lambda1;
        parameters.lambda2 = lambda2; 
        
        model_para = kMTRL_active_learning_Wb(trainX, trainY,init,Omega_truth, pair, Options,parameters);
        
        models{i} = model_para;
        [RMSE(i),b]   = make_evaluation(validX, validY, model_para.W,model_para.b);
         [RMSE_train(i), b] = make_evaluation(trainX, trainY, model_para.W,model_para.b);
        final_func(i) = model_para.func_all(end);
end
[~, index] = min(RMSE);
model = models{index};

output = struct();
output.model = model;
output.RMSE = RMSE;
output.index = index;
output.RMSE_train = RMSE_train;
output.final_func = final_func;
end