% clear;
% clc;

data_load = load(results_name);
 
feature_dims = data_load.feature_dims;
rmse_mean = data_load.rmse_mean;
rmse_std  = data_load.rmse_std;

norm_mean = data_load.norm_mean;
norm_std  = data_load.norm_std;

%%% plot first 100 features:
feature_dims = feature_dims(:,1:9);
rmse_mean    = rmse_mean(:,1:9);
norm_mean    = norm_mean(:,1:9);

%%% Plot rmse figure
linesymbol = {'--*','-.+'};
figure('Position',[50,50,600,500])
for i = 1:2
    plot(feature_dims,rmse_mean(i,:),linesymbol{i},'MarkerSize',10,'LineWidth',3);
    hold on;
end
xlabel('\fontsize{20}Dimension of features','FontWeight','bold');
ylabel('\fontsize{20}RMSE','FontWeight','bold');
legend({'MTRL','eMTRL'},...
    'FontSize', 20, 'FontWeight','bold','Location','east');
set(gca,'fontsize',12,'fontweight','bold');
print(gcf, sprintf('../results/MTRLvskMTRL_featvarys_rmse'),'-dpdf');

%%% Plot norm figure
figure('Position',[50,50,600,500])
for i = 1:2
    plot(feature_dims,norm_mean(i,:),linesymbol{i},'MarkerSize',10,'LineWidth',3);
    hold on;
end
 
xlabel('\fontsize{20}Dimension of features');
ylabel('\fontsize{20}Norm');
legend({'MTRL','eMTRL'},...
    'FontSize', 20, 'FontWeight','bold','Location','east');
set(gca,'fontsize',12,'fontweight','bold');
print(gcf, sprintf('../results/MTRLvskMTRL_featvarys_norm'),'-dpdf');