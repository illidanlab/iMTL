function model=MTRL_RR(Km,label,task_index,insIndex,ins_num)
    n=size(Km,1);
    task_num=length(insIndex);
    M12=zeros(n,task_num);
    for i=1:task_num
        M12(insIndex{i},i)=1;
    end
    tmp=[label,zeros(1,task_num)]/([[Km+diag(ins_num(task_index))/2,M12];[M12',zeros(task_num)]]);
    model.alpha=tmp(1:n);
    model.b=tmp((n+1):(n+task_num));
    clear tmp M12;
    clear Km label task_index insIndex ins_num;