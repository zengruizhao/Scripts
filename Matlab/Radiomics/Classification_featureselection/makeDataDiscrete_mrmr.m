function dataw_discrete=makeDataDiscrete_mrmr(dataw)
% dataw=training_set;
t=mean(dataw);
tt=repmat(t,size(dataw,1),1);
dataw_discrete=uint8(dataw>tt);
end
