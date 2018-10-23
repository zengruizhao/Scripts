function blockfeat = BinHOGFeature( b_mag,b_orient,cell_size,nblock,...
    bin_num, weight_vote)
% ����1��block��hog
% weight_vote: �Ƿ���и�˹��ȨͶƱ
 
% block��HOGֱ��ͼ
blockfeat=zeros(bin_num*nblock^2,1);
 
% ��˹Ȩ��
gaussian_weight=fspecial('gaussian',cell_size*nblock,0.5*cell_size*nblock);
 
% �ָ�block
for n=1:nblock
    for m=1:nblock
        % cell�����Ͻ�����
        x_off = (m-1)*cell_size+1;
        y_off = (n-1)*cell_size+1;
 
        % cell���ݶȴ�С�ͷ���
        c_mag=b_mag(y_off:y_off+cell_size-1,x_off:x_off+cell_size-1);
        c_orient=b_orient(y_off:y_off+cell_size-1,x_off:x_off+cell_size-1);
 
        % cell��hogֱ��ͼ
        c_feat=zeros(bin_num,1);
        for i=1:bin_num
            % �Ƿ���и�˹��Ȩ ͶƱ
            if weight_vote==false
                c_feat(i)=sum(c_mag(c_orient==i));
            else
                c_feat(i)=sum(c_mag(c_orient==i).*gaussian_weight(c_orient==i));
            end
        end
 
        % �ϲ���block��HOGֱ��ͼ��
        count=(n-1)*nblock+m;
        blockfeat((count-1)*bin_num+1:count*bin_num,1)=c_feat;
    end
end
% ��һ�� L2-norm
sump=sum(blockfeat.^2);
blockfeat = blockfeat./sqrt(sump+eps^2);