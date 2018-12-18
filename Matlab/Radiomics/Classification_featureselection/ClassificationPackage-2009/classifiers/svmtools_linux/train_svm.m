function train_svm( name , training_set , training_labels , c , g ,kernel)
    
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

    svmlwrite(  [ name '_training.data' ] , training_set , training_labels );
    
    [ a b ] = unix( [ abs_path 'svm-scale -s ' name '_training.range ' name '_training.data > ' name '_training.scale'] );
    
	cmd = [ abs_path 'svm-train -t ' num2str(knl) ' -c ' num2str(c) ' -g ' num2str(g) ' ' name '_training.scale ' name '_training.model' ];

	[ a b ] = unix( cmd );
