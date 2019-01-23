function [outputvolume, standardization_map] = int_stdn_landmarks_multiTemplate(inputvolume, templatedata, templvol_lm, varargin)
%[OUTPUTVOLUME,STANDARDIZATION_MAP] = int_stdn_landmarks(INPUTVOLUME,TEMPLATEVOLUME,OPTIONS)
%Landmark-based Intensity Standardization
%Scales data piecewise to a template, using histogram landmarks at every 10th percentile
%Note: Landmarks are automatically calculated from the data, making this step
%almost completely automatic
%Keep in mind the method is theoretically "finding" landmarks that should be matched up between
%input and template. The OPTIONS struct below gives you a way to pass in a mask for regions
%you want to IGNORE when calculating landmarks ALONE. Standardization is always 
%applied to the entire INPUTVOLUME.
%INPUTVOLUME - data to be standardized (NROW x NCOL x NSLICES)
%TEMPLATEVOLUME - template data (NROW x NCOL x NSLICES)
%Please make sure these are both integer-valued arrays (no decimals) else
%this program will ROUND them prior to processing
%
%OPTIONS is a STRUCT that can have any of the following fields:
%1. INCANCERMASKS,TEMCANCERMASKS - binary cancer regions (if available) (NROW x NCOL x NSLICES)
%Note: If *CANCERMASKS is supplied, these values are IGNORED when
%calculating STANDARDIZATION_MAP (i.e. no landmarks will be attached to those regions).
%However these areas WILL BE INCLUDED when calculating the final OUTPUTVOLUME (they "go along
%for the ride").
%2. NUMSTDTOPCLIP, NUMSTDBOTCLIP - '0' if no clipping, else how many std devs away you want to clip extreme histogram values 
%(default: NUMSTDTOPCLIP = 5, NUMSTDBOTCLIP = 0, "best" contrast)
%If both values passed in, will clip from top and bottom extremes of histogram
%Note: Try to leave NUMSTDTOPCLIP = NUMSTDBOTCLIP
%3. ZEROVAL - "zero" valued regions you want to ignore in your image, defaults to 0
%for e.g. everything outside a segmented prostate region would have value 0
%If set, everything <= ZEROVAL are IGNORED for the purposes of standardization
%4. DORESCALE - True (default) so that both INPUTVOLUME AND TEMPLATEVOLUME are rescaled 
% between 0-4095 by default
%If the data you're passing in has ALREADY been rescaled, please set this to false
%5. DOCHECK - True (default) will mean the program will popup a window to
%confim that the histograms are actually misaligned and you want to
%continue. Set to false if you don't want to keep clicking 'Yes'.
%
% Functions needed: count (plotdist), topclip, botclip, rescale_range
%
% SEE ALSO: int_stdn.m, plotdist.m, calcstdnmap.m, applystdnmap.m

%Based off original code by senior design group (2010-2011) -
%'data_scaler.m','standardizedprostate.m'
%TBD: put code here to remove a landmark if it's greater than the 99th
%percentile
%Satish Viswanath, Jan 2012
%Edits, Jun 2012: 
%- Moved to using OPTIONS struct instead of input variables
%- Additional bugfixes and error catchin for 2D data
%- Removed pointless end-prompt for 'Standardization Ok?'
%- Bugfix to handle non-integer inputs
%Edits, Jan 2013: Additional bugfixes
%Edits, May 2013: 
%- Rescale option
%- Modularized code by extracting steps into different functions

if nargin > 3
    options = varargin{1};
else
    options = [];
end

if ~isfield(options,'incancermasks')
%     disp('No cancer masks for input non-standardized data.'); 
	incancermasks = [];
else
    incancermasks = options.incancermasks;
end

if ~isfield(options,'numstdtopclip')
%     disp('Data values WILL be top-clipped (improves contrast).'); 
	numstdtopclip = 5;      
else
    numstdtopclip = options.numstdtopclip;
end
if logical(numstdtopclip) && logical(~exist('topclip','file'))
	error('Get TOPCLIP into your path!');
end 

if ~isfield(options,'numstdbotclip')
%     disp('Data values WILL NOT be bottom-clipped.'); 
	numstdbotclip = 0;
else
    numstdbotclip = options.numstdbotclip;
    if logical(~exist('botclip','file'))
        error('Get BOTCLIP into your path!');
    end  
end

if ~isfield(options,'zeroval')
    zeroval = 0;
else 
    zeroval = options.zeroval;
end

if ~isfield(options,'dorescale')
    dorescale = true;
else
    dorescale = options.dorescale;
end
if logical(dorescale) && logical(~exist('rescale_range','file'))
	error('Get RESCALE_RANGE into your path!');
end 

if ~isfield(options,'rescaleMax')
    rescaleMax = 4095;
else
    rescaleMax = options.rescaleMax;
end

checkifInt = find(inputvolume); checkifInt = inputvolume(checkifInt(1));
if ceil(checkifInt) ~= floor(checkifInt)    
%     disp('Rounding non-integer values in INPUTVOLUME');
    inputvolume = round(inputvolume);
end


%Linear rescaling of intensity ranges
if dorescale
    inputvolume = round(rescale_range(inputvolume, 0, rescaleMax));
end

%Sorting,vectorizing
inputdata=sort(inputvolume(:));

%Remove zeros
inputdata(inputdata<=zeroval)=[];

%Remove cancer masks
indata_forlm = inputvolume; indata_forlm(incancermasks)=[];
indata_forlm(indata_forlm<=zeroval)=[];

%Vectorize if not already vector (can happen for 2D images)
if length(find(size(indata_forlm)-1)) > 1
    indata_forlm = indata_forlm(:);
end

%Find landmarks at every 10th percentile
inputvol_lm = prctile(indata_forlm, (10:10:90));
landmarks = [inputvol_lm', templvol_lm'];

%Calculate standardization_map
standardization_map = calcstdnmap(inputdata, templatedata, landmarks);

%Apply standardization_map to standardize the inputvolume
outputvolume = applystdnmap(inputvolume, standardization_map);

%Clip outliers coz they mess up the image
if logical(numstdtopclip)
    outputdata=sort(outputvolume(:));outputdata(outputdata<=zeroval)=[];
    if ~isempty(find(outputvolume > median(outputdata)+numstdtopclip*std(outputdata), 1))
%         disp('Clipping a little to improve contrast..')
        outputvolume = topclip(outputvolume,[],numstdtopclip);
        if logical(numstdbotclip)
            if ~isempty(find(outputvolume < median(outputdata)-numstdbotclip*std(outputdata), 1))
                outputvolume = botclip(outputvolume,[],numstdbotclip);
            end
        end
        %outputdata=sort(outputvolume(:));outputdata(outputdata==0)=[];
    end
end
end

