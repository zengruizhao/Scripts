function blockfeat = BinHOGFeature( b_mag,b_orient,cell_size,nblock,...
    bin_num, weight_vote)
% 计算1个block的hog
% weight_vote: 是否进行高斯加权投票
 
% block的HOG直方图
blockfeat=zeros(bin_num*nblock^2,1);
 
% 高斯权重
gaussian_weight=fspecial('gaussian',cell_size*nblock,0.5*cell_size*nblock);
 
% 分割block
for n=1:nblock
    for m=1:nblock
        % cell的左上角坐标
        x_off = (m-1)*cell_size+1;
        y_off = (n-1)*cell_size+1;
 
        % cell的梯度大小和方向
        c_mag=b_mag(y_off:y_off+cell_size-1,x_off:x_off+cell_size-1);
        c_orient=b_orient(y_off:y_off+cell_size-1,x_off:x_off+cell_size-1);
 
        % cell的hog直方图
        c_feat=zeros(bin_num,1);
        for i=1:bin_num
            % 是否进行高斯加权 投票
            if weight_vote==false
                c_feat(i)=sum(c_mag(c_orient==i));
            else
                c_feat(i)=sum(c_mag(c_orient==i).*gaussian_weight(c_orient==i));
            end
        end
 
        % 合并到block的HOG直方图中
        count=(n-1)*nblock+m;
        blockfeat((count-1)*bin_num+1:count*bin_num,1)=c_feat;
    end
end
% 归一化 L2-norm
sump=sum(blockfeat.^2);
blockfeat = blockfeat./sqrt(sump+eps^2);