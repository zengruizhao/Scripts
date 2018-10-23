function [  ] = temp1( slide, cell, point, i, j, xmin, ymin, width, height, xmlFile, k, filetypeJpg )

stage = BOPointInPolygon(cell, point);%%%%%%%%%%%%%
if (stage == 'i') %||stage== 'o'
    patch = openslide_read_region(slide, xmin+ (j-1)*width, ymin + (i-1) * height, width, height, 0);
    II=patch(:,:,2:4);
    JPath=['/media/zzr/Data/skin_xml/',xmlFile(k).name(1:end-4),'/'];
    if ~exist(JPath)
        mkdir(JPath);
    end
    JpgPath=[JPath,xmlFile(k).name(1:end-4),'_',num2str(j),'_',num2str(i),filetypeJpg];
    imwrite(II,JpgPath);
end

