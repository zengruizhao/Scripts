function [ c_value , gamma_value ] = cv_svm( train_pathname , training_data , training_labels, kernel)

switch kernel
    case {0,'linear'}
        knl = 0;
    case {1,'poly'}
        knl = 1;
    case {2,'RBF','rbf'}
        knl = 2;
    case {3,'sigmoid'}
        knl = 3;
end

abs_path = '/data/common/code/Ajay/libsvm-2.89/./';

svmlwrite(  [ train_pathname '_cv.data' ] , training_data , training_labels );

scale_command = [ abs_path 'svm-scale -s ' train_pathname '_cv.range ' train_pathname '_cv.data > '  train_pathname '_cv.scale' ];
[status]=unix( scale_command );
if status,
    error('didn''t run svm-scale');
end

best_accuracy = -1 ;
second_best = -1;
third_best = -1;
best_c = -1;
best_g = -1;
best_second_c = -1;
best_second_g = -1;
best_third_c = -1;
best_third_g = -1;

fid = fopen([train_pathname '_results.txt'] ,'W');

for c = -5:2:10
    for g = -10:2:3
        cmd = [ abs_path 'svm-train -t ' num2str(knl) ' -c ' num2str(2^c) ' -g ' num2str(2^g) ' -v 3 ' train_pathname '_cv.scale' ];
        [ status , result ] = unix( cmd );
        
        result = result( (strfind( result , 'Accuracy' )-1):length(result)-2 );
        
        eval( [result ';'] );
        
        fprintf(fid, '%2f %2f %2f \n' , Accuracy , 2^c , 2^g );
        
        if( Accuracy > best_accuracy )
            third_best = second_best;
            second_best = best_accuracy;
            best_accuracy = Accuracy;
            
            best_third_c = best_second_c;
            best_third_g = best_second_g;
            
            best_second_c = best_c;
            best_second_g = best_g;
            
            best_c = c;
            best_g = g;
            
        end
    end
end
fclose( fid );

second_gamma_value = 2^best_second_g;

third_gamma_value = 2^best_third_g;

c_value = [ 2^best_c 2^best_second_c  2^best_third_c ] ;
gamma_value = [ 2^best_g 2^best_second_g  2^best_third_g ] ;
% fprintf( '1st %2f %2f %2f \n' , best_accuracy , c_value(1) , gamma_value(1) );
% fprintf( '2nd %2f %2f %2f \n' , second_best   , c_value(2) , gamma_value(2) );
% fprintf( '3rd %2f %2f %2f \n' , third_best    , c_value(3) , gamma_value(3) );
