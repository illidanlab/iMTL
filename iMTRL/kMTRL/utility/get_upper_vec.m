function z = get_upper_vec(X)
% This function get the upper triangular part of matrix
% and concatenate it to a vector row by row.


[m,n] = size(X);

z  = zeros(n*(n-1)/2,1);


p = 1;
for i = 1:m-1
    cnp = n - i;
    z(p:p+cnp-1) = X(i,i+1:m);
    p = p+cnp;
end



% 
% Xt = X';
% 
% sparsity_pattern = ~logical(triu(Xt));
% z  = Xt(sparsity_pattern);


end