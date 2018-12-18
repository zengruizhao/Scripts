function prediction = predict_svm( train_name , testing_data )

svmlwrite(  [ train_name '_testing.data' ] , testing_data  );

[ a b ] = dos( ['svmscale -r ' train_name '_training.range ' train_name '_testing.data > ' train_name '_testing.scale'  ] );
[ a b ] = dos( ['svmpredict '  train_name  '_testing.scale ' train_name '_training.model ' train_name '_testing.predict'] );

prediction = dlmread( 'decision_values.txt' , ' ');
%sign_prediction = dlmread( [ name '_testing.predict'] , ' ');


