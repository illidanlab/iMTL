function [model,predictions]= MTRL_Omegatrue(init, data,label,kertype,kerpar,lambda1,lambda2,testdata)
%Input:
%      data: a 1*m cell array. Each cell in 'data' contains the data matrix for one task where each row represents a data point for this task.
%      label: a 1*m cell array. Each cell in 'label' contains the labels for all data points of one task in a row vector.
%      kertype: a string. It is the kernel type with three candidaate options: 'linear', 'poly', 'rbf'.
%      kerpar: a scalar. It represents the kernel parameters with the format specified in function 'KernelFunction' in file 'CalculateKernelMatrix.m'.
%      lambda1 & lambda2: two positive scalars. They are the regularization parameters in the objective function.
%      testdata: a 1*m cell array. Each cell contains a test data matrix for one task with one row as one test data point.
%Output:
%      model: a struct variable. It contains the training model, which includes many fields, such as 
%                                'alpha', the dual variables,
%                                'b', the offset of the predictive function,
%                                'Omega', the task covariance matrix.
%      predictions: a 1*m cell array when 'testdata' is not empty. Each cell contains a row vector, which is the predictive function values for one task.

    %%% Kaixiang begin
    data_old = data;
    label_old = label;
    K=length(data);%task number
    for i = 1:K
    yy = label_old{i};
    label_old{i} = yy';  
    end

    func_all = [];
    
    % initialization
 
    
    %%% Kaixiang end
 

    m=length(data);%task number
    Omega=init.Omega_true;                                                     %%% Kaixiang 
    [data,label,task_index,ins_num]=PreprocessMTData(data,label); % data m*n x 1 vector [task1 samples;task2 samples; task3 samples]  lable [task1 labels,task2 labels,task3 labels] 
    n=size(data,1); % number of all samples.
    insIndex=cell(1,m);
    ins_indicator=zeros(m,n);
    for i=1:m
        insIndex{i}=sort(find(task_index==i));
        ins_indicator(i,insIndex{i})=1;
    end
    max_iteration=1;                                                     %%% Kaixiang 
    threshold=10^(-5);                                                     %%% Kaixiang 
    model.alpha=zeros(1,n);                                                     %%% Kaixiang 
    model.b=zeros(1,m);                                                     %%% Kaixiang 
    Km=CalculateKernelMatrix(data,kertype,kerpar); % data*data'
    epsilon=10^(-8);
%     m_Cor= real(Omega/(lambda1*Omega+lambda2*eye(m))); % Omega(lambda_1*Omega + lambda_2*Im)^{-1} %%% Kaixiang 
    m_Cor= real(pinv(lambda1*eye(m) + lambda2*pinv(Omega)));
    %%% Kaixiang begin
%     old_W = CalculateW(data,model,task_index,m,Omega,lambda1,lambda2);
    %%% Kaixiang end
    
    for t=1:max_iteration
        old_model=model;
%         old_Omega=Omega;                                                      %%% Kaixiang 
        %Calculate alpha and b
        MTKm=Km.*m_Cor(task_index,task_index);
        model=MTRL_RR(MTKm,label,task_index,insIndex,ins_num); % Using linear equation to solve the model.
        clear MTKm;
        
        %%% Kaixiang 
        %Calculate Omega
%         temp=m_Cor(:,task_index)*diag(model.alpha);
%         temp=temp*Km*temp';
%          
%         [eigVector,eigValue]=eig(temp+epsilon*eye(m));
%         clear temp;
%   
% %         eigValue=sqrt(abs(diag(eigValue)));
%         eigValue=eigValue/sum(eigValue);
%         Omega=eigVector*diag(eigValue)*eigVector';                                                   %%% Kaixiang 
%         m_Cor=eigVector*diag(eigValue./(lambda1*eigValue+lambda2))*eigVector';    %%% Kaixiang         % Omega(lambda_1*Omega + lambda_2*Im)^{-1} plugin eigen decomposition.
%         clear eigVector eigValue;
                                                                           %%% Kaixiang 
%         if norm(model.alpha-old_model.alpha,2)<=threshold*n&&norm(model.b-old_model.b,2)<=threshold*m 
%             clear old_model  ;
%             break;
%         end
%         clear old_model old_Omega;
        
        %%% Kaixiang begin
        W = CalculateW(data,model,task_index,m,Omega,lambda1,lambda2);
        funcval = func_val(data_old, label_old, Omega,lambda1,lambda2, W,model.b);
        func_all = cat(1, func_all, funcval);
        
%         if norm(W - old_W, 'fro') <= threshold*m*size(W,1) 
% %             clear old_model old_Omega;
%             break;
%         end
        
        
%         old_W = W;
        
        %%% Kaixiang begin
    end
    
    if strcmp(kertype,'linear')==1        
        model.W=CalculateW(data,model,task_index,m,Omega,lambda1,lambda2);
        clear W;
    end
    model.Omega=Omega;
    model.lambda1=lambda1;
    model.lambda2=lambda2;
    
    %%% Kaixiang begin
    model.func_all = func_all;
    %%% Kaixiang begin
    
    clear WSquare Omega;
    if nargin==7
        predictions=[];
    else
        predictions=cell(1,m);
        for i=1:m
            if isempty(testdata{i})
                continue;
            end
            predictions{i}=Testing(testdata{i},i,data,task_index,model,kertype,kerpar,m_Cor);
        end
    end
    clear data label Omega Km task_index insIndex;
    
function prediction=Testing(testdata,task_no,data,task_index,model,kertype,kerpar,m_Cor)
    alpha=model.alpha;
    b=model.b(task_no);
    index=find(alpha~=0);
    test_Km=CalculateKernelMatrix(data(index,:),testdata,kertype,kerpar);
    prediction=(alpha(index).*m_Cor(task_no,task_index(index)))*test_Km+b;
    clear testdata task_no alpha index data task_index model kertype kerpar m_Cor;
    
function W=CalculateW(data,model,task_index,m,Omega,lambda1,lambda2)
    I=eye(m);
    W=zeros(size(data,2),m);
    for i=1:size(data,1)
        W=W+model.alpha(i)*data(i,:)'*I(task_index(i),:);
    end
    W=W*real(Omega*pinv(lambda1*Omega+lambda2*eye(m)));                         %%% Kaixiang 
    clear data model;