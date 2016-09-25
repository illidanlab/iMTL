function z = get_upper_vec(X)
% This function get the upper triangular part of matrix
% and concatenate it to a vector row by row.

Xt = X';

sparsity_pattern = ~logical(triu(Xt));
z  = Xt(sparsity_pattern);

end