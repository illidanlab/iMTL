function [Omega_label,num_constraints_tasks] = label_pairs(pairs,task_num)
% this function label the selected constraints, if select (1,2) and (3,4)
% then Omega_label(1,2) += 1; Omega_label(3,4) += 1; 

Omega_label = zeros(task_num);

num_pairs = size(pairs,2);

num_constraints_tasks = zeros(task_num,1);

for i = 1:num_pairs
    
    i1 = pairs(1,i);
    j1 = pairs(2,i);
    i2 = pairs(3,i);
    j2 = pairs(4,i);
    
    Omega_label(i1,j1) = Omega_label(i1,j1) + 1;
    Omega_label(i2,j2) = Omega_label(i2,j2) + 1;
    Omega_label(j1,i1) = Omega_label(i1,j1);
    Omega_label(j2,i2) = Omega_label(i2,j2);
    
    for j =1:4
        task_i = pairs(j,i);
        num_constraints_tasks(task_i) = num_constraints_tasks(task_i) +1;
    end
        
end

 

end