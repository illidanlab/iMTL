function [row_id, column_id] = decode_triangular_id(triangular_id,num)
% this function decode the index of element in triangular matrix in a size of 
% num x num matrix,
% to row id
% and column id.

      temp = triangular_id;
      curr_row = 1;
      while temp >0
          temp_num = num - curr_row;
          temp = temp - temp_num;
          
          curr_row = curr_row + 1;
      end
      row_id = curr_row-1;
      column_id = num+ temp;




end
