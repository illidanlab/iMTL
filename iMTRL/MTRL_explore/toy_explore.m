   
clear;
addpath(genpath('./'));
%% Set parameters
num = 5;
noise_level = 10;
sample_energe = 1 ;
WT = [[3,10];       
      [-2,5];
      [10,1];
      ];
rng(1024);
noise_radio = [3,1,1];

%% Generate Data
W = WT';

[feature_dim, task_number] = size(W);
feature_dim = feature_dim - 1;

data  = cell(1,task_number);
label = cell(1,task_number);
    
    
    
for i = 1:task_number
   rng(1024)
   X = rand(num,feature_dim)*sample_energe;
   y = rand(num,1)*noise_level*noise_radio(i);

   w = W(:,i);
   y = y + X*w(1:end-1) + w(end);

   label{i} = y';
   data{i}  = X;

end

W_real = W(1:end-1,:);
Omega_real = (W_real'*W_real)^(1/2);
Omega_real = real(Omega_real/trace(Omega_real));    
    
%% Train Model MTRL 
tic;
    model=MTRL(data,label,'linear',0,0.01,0.05);
toc;

Omega_real;
Omega_learn = real(model.Omega);

corr_real = corrcov(Omega_real);
corr_learn = corrcov(real(model.Omega));

MTRL_W = real(model.W);
    
 %% kMTRL
K = task_number;
label_new = cell(K,1);
for i = 1:K
    yy = label{i};
    label_new{i} = yy';  
end

configureFile_toy_explore; 


model_true = kMTRL_Omega_true_Wb_interface(data, label_new,Omega_real,W_real,data,label_new, configurePara.kMTRL_Omega_true);

kMTRL_W = real(model_true.W);

%% display
disp('The real correlation matrix')
disp(corr_real)

disp('The correlation matrix by learned Omega')
disp(corr_learn)

disp('W from MTRL')
disp(MTRL_W)
disp('l2 norm of the MTRL_W - W_real')
disp(norm(MTRL_W-[3,-2,10], 'fro'))

disp('W from kMTRL')
disp(kMTRL_W)
disp('l2 norm of the kMTRL_W - W_real')
disp(norm(kMTRL_W-[3,-2,10], 'fro'))
 
    