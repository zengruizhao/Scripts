function plotEigV(EigV,labels,titlestring,methodstring,method,k,plots);
if plots == 1
    figure;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% One Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:max(labels)
        subset=(labels==i);
        plot(EigV(subset,1),EigV(subset,2),PlotColour(i));hold on
        xlabel('1st Component');
        ylabel('2nd Component');

        if method == 5 |method ==6
            title([methodstring ' on ' titlestring ' k = ' num2str(k)]);
        else
            title([methodstring ' on ' titlestring ]);
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

if plots == 2
    figure;
    for i = 1:max(labels)
        subset=(labels==i);
        subplot(2,2,1)
        plot(EigV(subset,1),EigV(subset,2),PlotColour(i));hold on
        xlabel('1st Component');
        ylabel('2nd Component');
        %     legend('tumor','normal');
        %     legend('relapse','nonrelapse');
        %     legend('Mesothelioma (MPM)', 'adenocarcinoma (ADCA)');
        %     legend('BCR-ABL', 'E2A-PBX1', 'Hyperdiploid>50', 'MLL', 'T-ALL', 'TEL-AML1', 'Others');
        %     legend('ALL','AML');
        %     legend('ALL','MLL','AML');
        %     legend('survivor','failure');
        subplot(2,2,2)
        plot(EigV(subset,2),EigV(subset,3),PlotColour(i));hold on
        xlabel('2nd Component');
        ylabel('3rd Component');
        subplot(2,2,3)
        plot(EigV(subset,2),EigV(subset,4),PlotColour(i));hold on
        xlabel('2nd Component');
        ylabel('4th Component');
        subplot(2,2,4)
        plot(EigV(subset,3),EigV(subset,4),PlotColour(i));hold on
        xlabel('3rd Component');
        ylabel('4th Component');
    end

    if method == 5 |method ==6
        subplot_title([methodstring ' on ' titlestring ' k = ' num2str(k)]);
    else
        subplot_title([methodstring ' on ' titlestring ]);
    end
    %subplot_title2('Data obtained from http://sdmc.lit.org.sg/GEDatasets/Datasets.html');
end

if plots == 3
    figure;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%3D plot%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:max(labels)
        subset=(labels==i);
        plot3(EigV(subset,1),EigV(subset,2),EigV(subset,3),PlotColour(i));hold on
        xlabel('1st Component');
        ylabel('2nd Component');
        zlabel('3rd Component');
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

