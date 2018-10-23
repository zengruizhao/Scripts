function cellbw_out  = reserve_circle_region(cellbw_in)

[cellbw,num] = bwlabel(cellbw_in,4);
cellbw_out = regionprops(cellbw,'Area');
cellbw_Major = regionprops(cellbw,'MajorAxisLength');
cellbw_Minor = regionprops(cellbw,'MinorAxisLength');
for i = 1 : num
   region_Major(i,1) = cellbw_Major(i).MajorAxisLength;
end
for i = 1 : num
   region_Minor(i,1) = cellbw_Minor(i).MinorAxisLength;
end
divide = region_Major./region_Minor;
 region_det = find(divide>=3);%≥§∂Ã÷·±»¬ 
 cellbw_out = cellbw;
 
for i =1 : length(region_det)
   cellbw_out(find(cellbw ==region_det(i))) = 0;
end
cellbw_out = logical(cellbw_out);

end