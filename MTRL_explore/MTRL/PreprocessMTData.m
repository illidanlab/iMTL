function [data,label,task_index,ins_num]=PreprocessMTData(data,label)
    m=length(data);
    newdata=[];
    newlabel=[];
    task_index=[];
    ins_num=zeros(1,m);
    for i=1:m
        newdata=[newdata;data{i}];
        newlabel=[newlabel,label{i}];
        task_index=[task_index,i*ones(1,size(data{i},1))];
        ins_num(i)=size(data{i},1);
    end
    clear data label;
    data=newdata;
    label=newlabel;
    clear newdata newlabel;