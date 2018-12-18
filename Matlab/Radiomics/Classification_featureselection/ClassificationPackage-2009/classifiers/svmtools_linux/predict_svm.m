function prediction = predict_svm( train_name , testing_data )

abs_path = '/data/common/code/Ajay/libsvm-2.89/./';

svmlwrite(  [ train_name '_testing.data' ] , testing_data  );

[ a b ] = unix( [ abs_path 'svm-scale -r ' train_name '_training.range ' train_name '_testing.data > ' train_name '_testing.scale'  ] );

[ a b ] = unix( [ abs_path 'svm-predict '  train_name  '_testing.scale ' train_name '_training.model ' train_name '_testing.predict'] );

%prediction = dlmread( 'decision_values.txt' , ' ');
prediction = dlmread( [ train_name '_testing.predict'] , ' ');
