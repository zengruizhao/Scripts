clear;
clc;
addpath(genpath('../../openslide-matlab-master'));
addpath('./BOPointInPolygon')
ImgPath= '../test_data/';
xmlPath= '../test_data/';
xmlFile = dir([xmlPath,'*.xml']);

openslide_load_library();
for k=1:length(xmlFile)
    oneXml = [xmlPath,xmlFile(k).name];
    c = parseCamelyon16XmlAnnotations_ccf(oneXml);
%     WSI = [ImgPath,xmlFile(k).name(1:end-4),'.ndpi'];
    WSI = [ImgPath '/2018-06-07 17.02.07.ndpi'];
    slide = openslide_open(WSI);
    LL = size(c);
    LL = LL(1,1);
    for kk=1:LL
        cell=c{kk,1};
        xmin=min(cell(:,1));
        ymin=min(cell(:,2));
        xmax=max(cell(:,1));
        ymax=max(cell(:,2));
        Img=openslide_read_region(slide,xmin,ymin,xmax-xmin,ymax-ymin,0);
        I=Img(:, :, 2:4);
        %         figure(2);
        %         imshow(I,[]);
        [w, h, ~] = size(I);
        height = 400;%256
        width = 400;
        max_row = floor(h / height);
        max_col = floor(w/ width);
        %         Lastheight = (h - max_row * height) ;
        %         Lastwidth = (w - max_col * width);
        filetypeJpg = '.png';
        for i=1:max_row
            for j=1:max_col
                pointx = xmin+ j * width - 200; %128
                pointy = ymin+ i * height - 200;
                P = c{1,1}(1,:);
                PP = [cell;P];
                point = [pointx,pointy];
                stage = BOPointInPolygon(PP, point);
                disp(['Stage: ' stage]);
                if (stage == 'i') %||stage== 'o'
                    patch = openslide_read_region(slide, xmin + j * width, ymin + i * height, 400, 400, 0);
                    II=patch(:,:,2:4);
                    JPath=['/home/xjw/Projects/Cholangiocarcinoma/iter9_ly_seg_clas/data/patches/',xmlFile(k).name(1:end-4),'/'];
                    if ~exist(JPath)
                        mkdir(JPath);
                    end
                    JpgPath=[JPath,xmlFile(k).name(1:end-4),'_',num2str(j),'_',num2str(i),filetypeJpg];
                    imwrite(II,JpgPath);
                end
            end
        end
    end
end
