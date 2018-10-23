clear;clc;
addpath(genpath('../../openslide-matlab-master'));
addpath('./BOPointInPolygon')
ImgPath= '/run/user/1000/gvfs/smb-share:server=darwin-mi,share=data/Skin/MF_10/20180627/';
xmlPath= '/run/user/1000/gvfs/smb-share:server=darwin-mi,share=data/Skin/MF_10/2018-06-06 16.14.09修改标1/';
xmlFile = dir([xmlPath,'*.xml']);
openslide_load_library();
for k=2%:length(xmlFile)
    fprintf([xmlFile(k).name '...\n']);
    oneXml = [xmlPath,xmlFile(k).name];
    c = parseCamelyon16XmlAnnotations_ccf(oneXml);
%     WSI = [ImgPath,xmlFile(k).name(1:end-4),'.ndpi'];
    WSI = [ImgPath '/2018-06-06 16.14.09.ndpi'];
    slide = openslide_open(WSI);
    [mppX, mppY, width, height, numberOfLevels, ...
    downsampleFactors, objectivePower] = openslide_get_slide_properties(slide);
    LL = size(c);
    LL = LL(1,1);
    for kk=1:LL
        fprintf('%dth area...\n', kk);
        cell=c{kk,1};
        xmin=min(cell(:,1));
        ymin=min(cell(:,2));
        xmax=max(cell(:,1));
        ymax=max(cell(:,2));
        Img=openslide_read_region(slide,xmin,ymin,xmax-xmin,ymax-ymin,0);
        I=Img(:, :, 2:4);
        [w, h, ~] = size(I);
        height = 224;
        width = 224;
        %% gurrante
        hh = max(height, h);
        ww = max(width, w);
        if w<width || h<height
             Img=openslide_read_region(slide, xmin, ymin, ww, hh, 0);
        end
        I=Img(:, :, 2:4);
        [w, h, ~] = size(I);
        %%
        if ~isempty(strfind(xmlFile(k).name, '汗腺')) idx = 16; end%16
        if ~isempty(strfind(xmlFile(k).name, '毛囊')) idx = 16; end%16
        if ~isempty(strfind(xmlFile(k).name, '表皮')) idx = 4; end%4
        if ~isempty(strfind(xmlFile(k).name, '脂肪')) idx = 2; end%%2
        if ~isempty(strfind(xmlFile(k).name, '真皮')) idx = 1; end%%1
        if ~isempty(strfind(xmlFile(k).name, '背景')) idx = 1; end%1
        %%
         for i=1:(height/idx):(h-height+1)
                for j=1:(width/idx):(w-width+1)
                    pointx = xmin +  width/2 + j-1; %
                    pointy = ymin+  height/2 + i-1;%
                    point = [pointx,pointy];
                    stage = BOPointInPolygon(cell, point);
                    if (stage == 'i') %||stage== 'o'
                        patch = openslide_read_region(slide, xmin + j -1, ymin + i-1, width, height, 0);
                        II=patch(:,:,2:4);
                        JPath=['/media/zzr/Data/skin_xml/',xmlFile(k).name(1:19), '/', xmlFile(k).name(1:end-4),'/'];
                        if ~exist(JPath)
                            mkdir(JPath);
                        end
                        JpgPath=[JPath, num2str(kk), '_', num2str(j), '_', num2str(i), '.png'];
                        imwrite(II,JpgPath);
                    end
                end      
         end
    end
end
