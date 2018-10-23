function HOGFeature = ImgHOGFeature( imgName, cell_size, nblock,...,
    overlap, angle, bin_num)
% ��������ͼ�������������
% imgPathName��ͼƬ·��
% cell_size: cell�ĳ���
% nblock: block��width��height������cell�ĸ���
% overlap: block�ص��ı���
% angle: 180\360
% bin_num: ����bin����Ŀ
 
if nargin<2
    % default parameter
    cell_size=7;
    nblock=3;
    overlap=0.5;
    angle=180;
    bin_num=9;
elseif nargin<6
    error('Input parameters are not enough.');
end
 
Img = imgName;
if size(Img,3) == 3
    % �򻯼��㣬ֱ��ת���ɻҶ�ͼ��
    G = rgb2gray(Img);
else
    G = Img;
end
 
[height, width] = size(G);
 
% ����x��y������ݶ�
hx = [-1,0,1];
hy = -hx';
grad_x = imfilter(double(G),hx);
grad_y = imfilter(double(G),hy);
 
% �����ݶȵ�ģ��
grad_mag=sqrt(grad_x.^2+grad_y.^2);
 
% �����ݶȵķ���
index= grad_x==0;
grad_x(index)=1e-5;
YX=grad_y./grad_x;
if angle==180
    grad_angle= ((atan(YX)+(pi/2))*180)./pi; 
elseif angle==360
    grad_angle= ((atan2(grad_y,grad_x)+pi).*180)./pi;
end
 
% orient bin
bin_angle=angle/bin_num;
grad_orient=ceil(grad_angle./bin_angle);
 
% ����block�ĸ���
block_size=cell_size*nblock;
skip_step=block_size*overlap;
x_step_num=floor((width-block_size)/skip_step+1);
y_step_num=floor((height-block_size)/skip_step+1);
 
% ��ʼ��hog����������
feat_dim=bin_num*nblock^2;
HOGFeature=zeros(feat_dim,x_step_num*y_step_num);
 
for k=1:y_step_num
    for j=1:x_step_num
        % block�����Ͻ�����
        x_off = (j-1)*skip_step+1;
        y_off = (k-1)*skip_step+1;
 
        % ȡ��block���ݶȴ�С�ͷ���
        b_mag=grad_mag(y_off:y_off+block_size-1,x_off:x_off+block_size-1);
        b_orient=grad_orient(y_off:y_off+block_size-1,x_off:x_off+block_size-1);
 
        % ��ǰblock��hogֱ��ͼ
        currFeat = BinHOGFeature(b_mag, b_orient, cell_size,nblock, bin_num, false);
        HOGFeature(:, (k-1)*x_step_num+j) = currFeat;
 
    end
end
 

 
