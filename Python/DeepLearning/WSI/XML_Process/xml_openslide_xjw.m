clear;
clc;
addpath('/home/xjw/Matlab_lib/openslide-matlab');
addpath('/home/xjw/brain/Area_Seg/Code/BOPointInPolygon/')
ImgPath = '/media/xjw/D6EA48CEEA48AD15/Oral Cancer/Multi-nucleation/Oroph_SCC/Original/svs/5_001.svs';
xmlPath = '/home/xjw/Desktop/5_001_g.xml';

c = parseCamelyon16XmlAnnotations_ccf(xmlPath);
WSI = ImgPath;
slide = openslide_open(WSI);
LL = size(c);
LL = LL(1,1);
for kk=1:LL
    cell = c{kk,1};
    xmin=min(cell(:,1));
    ymin=min(cell(:,2));
    xmax=max(cell(:,1));
    ymax=max(cell(:,2));
    Img = openslide_read_region(slide,xmin,ymin,xmax-xmin,ymax-ymin,'level',0);
    I = Img(:, :, 2:4);
    %         figure(2);
    %         imshow(I,[]);
    [w, h, ~] = size(I);
    height = 256;%256
    width = 256;
    max_row = floor(h / height);
    max_col = floor(w / width);
    %         Lastheight = (h - max_row * height) ;
    %         Lastwidth = (w - max_col * width);
    filetypeJpg = '.tif';
    for i=1:max_row
        for j=1:max_col
            pointx = xmin + j * width + 128; %128
            pointy = ymin + i * height + 128;
            P = c{1,1}(1,:);
            PP = [cell;P];
            point = [pointx,pointy];
            stage = BOPointInPolygon(PP, point);
            disp(['Stage: ' stage]);
            if (stage == 'i') %||stage== 'o'
                patch = openslide_read_region(slide, xmin + j * width, ymin + i * height, 256, 256, 'level', 0);
                II = patch(:,:,2:4);
                JPath=['/home/xjw/Desktop/Untitled Folder/area/img/','5_001/'];
                if ~exist(JPath)
                    mkdir(JPath);
                end
                JpgPath = [JPath, '5_001_', num2str(kk), '_', num2str(j),'_',num2str(i),filetypeJpg];
                imwrite(II, JpgPath);
             end
         end
     end
end