function [standardization_map] = calcstdnmap(indata, templdata, landmarks)
%[STANDARDIZATION_MAP] = calcstdnmap(INDATA,TEMPLDATA)
%Calculate standardization map for landmark-based intensity standardization
%INDATA AND TEMPLDATA must be vectors
%INDATA is the vector data that needs to be standardized
%TEMPLDATA is the template data vector to which INDATA is being
%standardized to
%LANDMARKS is a lm x 2 matrix of the form
%[input_distribution_landmarks,template_distribution_landmarks];
%where lm is the number of landmarks
%
% Satish Viswanath, May 2013
%
% SEE ALSO: int_stdn_landmarks.m, int_stdn.m

%Checks and balances
if ~isvector(indata) || ~isvector(templdata)
    error('INDATA, TEMPLDATA can only be vectors');
end

if ~ismatrix(landmarks)
    error('LANDMARKS should be a 2D matrix');
end

%Have to figure out the top and bottom ends of the standardization map
%separately
in_01=round(max(indata)*.01);
in_99=round(max(indata)*.99);
in_max=round(max(indata));
templ_01=round(max(templdata)*.01);

botperc=zeros(in_01,2);botperc(:,1)=(1:in_01)';
topperc=zeros(in_max-in_99+1,2);topperc(:,1)=(in_99:in_max)';

%Initialize the middle part of the standardization map
intensity_range=in_99-in_01; %range of the data
standmap = zeros(intensity_range+1,2);
standmap(:,1) = 1:intensity_range+1; %this is the input intensity, the value that is going to be standardized

%Now intialize the 'segments' within the map
L=size(landmarks);
segments = ones(L(1)+2,L(2));
segments(size(segments,1),1) = intensity_range+1;
segments(size(segments,1),2) = max(templdata)-min(templdata)+1;

%Set the minimum of 'segments' as 1
segments(1,1) = 1;
segments(1,2) = 1;

%Check for landmarks, modify segments accordingly
if numel(landmarks)>0
    segments(2:(L(1)+1),:)=landmarks;
end

%Scale within each 'segment'
for seg=1:size(segments,1)-1
    %Initialize
    pre_scaling = round(segments(seg,1):segments(seg+1,1))';
        
    %Scaling
    post_scaling = pre_scaling - min(pre_scaling(:));
    scalefactor=(segments(seg+1,2)-segments(seg,2))/(max(post_scaling)-min(post_scaling));
    post_scaling = post_scaling*scalefactor;
    post_scaling = post_scaling + segments(seg,2);        
    
    %Storing the scale factors to apply to the top and bottom ends of the
    %map later
    if seg==1
        scalefactends(1)=scalefactor;
    elseif seg==(size(segments,1)-1)
        scalefactends(2)=scalefactor;   
    end
    
    standmap(pre_scaling(1):pre_scaling(length(pre_scaling)),2)=post_scaling;
end

%Apply scale factors to the top and bottom ends of the map
botperc(:,2)=botperc(:,1)*scalefactends(1);
topperc(:,2)=topperc(:,1)*scalefactends(2);

%Add back the minimums
standmap(:,1)=standmap(:,1)+in_01-1;
%standmap(:,2)=standmap(:,2)+min(templatedata)-1;
standmap(:,2)=standmap(:,2)+templ_01-1;
standardization_map=[botperc;standmap;topperc];

%Round off any decimals
standardization_map = round(standardization_map);

end
