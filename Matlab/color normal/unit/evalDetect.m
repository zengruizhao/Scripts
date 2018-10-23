function [precision, recall, F1, matching,dot,gg,yy,rr, tp1, fp1, fn1,detect_tp_index,detect_fp_index] =...
    evalDetect(x, y, xGT, yGT, threshold, image,plot_pre_rec)
%Evaluation script to match detections based on the hungarian algorithm （评价的脚本，来评定检测的结果）
%OUTPUT
%   precision, recall, tp (#true positives), fp(#false positives, fn(#false negatives)
%INPUT
%   x,y,xGT,yGT - column vectors with x and y coordinates of detections and GT dots     
%   weightMap = used to recompute the tolerance threshold to accept detections
%       base on depthMaps
%   threshold = maximum distance between detection and GT dot to be matched.
%   image = (optional) argument to visualize the matches and misses. 

weightMap = ones(size(rgb2gray(image)));
depthMap = 1.0./sqrt(weightMap);

xGT_ = int32(xGT);
yGT_ = int32(yGT);

xGT_(xGT_ < 1) = 1;
yGT_(yGT_ < 1) = 1;
xGT_(xGT_ > size(weightMap,2)) = size(weightMap,2);
yGT_(yGT_ > size(weightMap,1)) = size(weightMap,1);

thresh = threshold*depthMap(sub2ind(size(depthMap),yGT_,xGT_));

dy = repmat(double(yGT), 1, size(y,1))- repmat(double(y)', size(yGT,1), 1);
dx = repmat(double(xGT), 1, size(x,1))- repmat(double(x)', size(xGT,1), 1);
dR = sqrt(dx.*dx+dy.*dy);
dR(dR > repmat(thresh, 1, size(y,1))) = +inf;
matching = Hungarian(dR);

fp1 = numel(x)-sum(matching(:));
fn1 = numel(xGT)-sum(matching(:));
tp1 = sum(matching(:));
precision = tp1 / (tp1 + fp1);
recall = tp1 / (tp1 + fn1);
F1=2*precision*recall/(precision+recall);

%% plot precision/recall
[x_mh,y_mh]=find(matching~=0);%x_mh表示groundtruth中的第几个，y_mh表示与之匹配的检测出数组中第几个
detect_tp_index=x_mh;
detect_fp_index=y_mh;

tp = zeros(size(x,1),1);%truth-positive
tp(y_mh) = 1;
fp = 1-tp;
cum_fp=cumsum(fp);
cum_tp=cumsum(tp);
rec=cum_tp/size(xGT,1);
prec=cum_tp./(cum_fp+cum_tp);
ap=VOCap(rec,prec);

indel_rec = meshgrid(1:5:size(rec,1));
indel_prec = meshgrid(1:5:size(rec,1));
if plot_pre_rec
    figure(12)
    plot(rec(indel_rec(1,:)),prec(indel_prec(1,:)),'-','LineWidth',2);
    axis([0 1 0 1])
    grid;
    xlabel('recall','FontSize',14);
    ylabel('precision','FontSize',14);
    title(sprintf('F1 = %.3f   Average Precision = %.3f',F1,ap),'FontSize',14);
    set(12, 'Color', [.988, .988, .988])
    pause(0.1) %let's ui rendering catch up
    average_precision_image = frame2im(getframe(12));
    imwrite(average_precision_image, 'visualizations/average_precision.png')
end
% figure(13)
% plot(cum_fp/sum(fp),rec,'-','LineWidth',2);
% axis([0 1 0 1])
% grid;
% xlabel('False positives rate','FontSize',14);
% ylabel('True positives rate(recall)','FontSize',14);
% title(sprintf('Pre:%1.4f   Rec:%1.4f   F1:%1.4f',precision,recall,F1),'FontSize',14);
% set(13, 'Color', [.988, .988, .988])
% pause(0.1) %let's ui rendering catch up
% ROC_image = frame2im(getframe(13));
% imwrite(ROC_image, 'visualizations/ROC.png')
% dot=[numel(y(any(matching))) numel(y(~any(matching))) numel(yGT(~any(matching,2)))];
%% plot points
if plot_pre_rec
figure,imshow(image); hold on;
% scatter(y(any(matching)),x(any(matching)),'g','o','fill');
scatter(y(any(matching)),x(any(matching)),'g','o','fill');
scatter(y(~any(matching)),x(~any(matching)),'y','o','fill');
scatter(yGT(~any(matching,2)),xGT(~any(matching,2)),'r','o','fill');
re.g = numel(y(any(matching)));
re.y = numel(y(~any(matching)));
re.r = numel(yGT(~any(matching,2)));
hold off;
xlabel(sprintf('Pre:%1.4f   Rec:%1.4f   F1:%1.4f',precision,recall,F1),'FontSize',14);
legend(['TP= ',num2str(re.g)],['FP= ',num2str(re.y)],['FN=',num2str(re.r)]);

disp(['Precision: ' num2str(precision) ' Recall: ' num2str(recall) ' F1: ' num2str(F1)]);

plotparam.precision = precision;
plotparam.recall = recall;
plotparam.F1 = F1;
plotparam.ap = ap;
plotparam.tp = tp;
plotparam.fp = fp;
plotparam.re = re;
plotparam.rec = rec;
plotparam.prec = prec;
plotparam.cum_fp = cum_fp;

gg=[y(any(matching)),x(any(matching))];
yy=[y(~any(matching)),x(~any(matching))];
rr=[yGT(~any(matching,2)),xGT(~any(matching,2))];
% save('plotparam/plotparam.mat','plotparam');

end
end
