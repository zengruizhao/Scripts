function write_data_set( name , train_set , train_labels , test_set , test_labels )

write_c45( [ name '.data' ] ,  train_set, train_labels );
write_c45( [ name '.test'  ]  ,  test_set, test_labels );
write_name_file( name  , train_set  ,train_labels);


function write_c45( name , data_set , data_labels )
fid = fopen( name , 'W');
for i = 1:size(data_set,1);
    for j = 1:size(data_set,2)
        fprintf(fid , '%d,' , data_set(i,j) ); 
    end
    
    fprintf(fid,'%d\n', data_labels(i,1));    
end
fclose( fid );

function write_name_file( name , data_set ,data_labels )

fid = fopen( [ name '.names' ] , 'W' ) ;
labels = unique(data_labels );

for i = 1:length(labels )-1 
    fprintf( fid , '%d,' , labels(i) );
end
fprintf( fid , '%d\n\n' , labels(length(labels)) );

for i = 1:size(data_set,2)
    fprintf( fid , '%i_feature: continuous.\n' , i ); 
end
fclose(fid);