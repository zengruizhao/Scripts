function [resbbox,resconf]=prunebboxes(bbox,conf,ovthresh)
  
[vs,is]=sort(-conf);
bbox=bbox(is,:);
conf=conf(is);

resbbox=[];
resconf=[];
rescount=0;
freeflags=ones(size(conf));
while sum(freeflags)>0
  indfree=find(freeflags);
  [vm,im]=max(conf(indfree));
  indmax=indfree(im);
  ov=bboxoverlapval(bbox(indmax,:),bbox(indfree,:));
  indsel=indfree(find(ov>=ovthresh));
  resbbox(rescount+1,:)=mean(bbox(indsel,:),1);
  resconf(rescount+1,:)=conf(indmax);
  rescount=rescount+1;
  freeflags(indsel)=0;
end
