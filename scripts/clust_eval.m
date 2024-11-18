% Function to plot the Ground Truth image and the first Principal Component vs the clustering results
function [] = clust_eval(Total_idx,algorithm,Z_total_cube,Salinas_Labels,existed_L) 
    figure;
    colormap('jet');
    subplot(2,2,1);
    imagesc(Salinas_Labels)
    axis off 
    title('Ground Truth')
    colorbar
    subplot(2,2,2)
    imagesc(Z_total_cube(:,:,1))
    axis off 
    title('PC1')
    count  = 3;
    for i = 1:(size(Total_idx,2))
        hold on
        cl_label = Total_idx(:,i);
        cl_label_tot=zeros(220*120,1);
        cl_label_tot(existed_L)=cl_label;
        im_cl_label=reshape(cl_label_tot,220,120);
        subplot(2,2,count);
        imagesc(im_cl_label)
        axis off
        title(strcat(algorithm,' | Clusters: ', int2str(length(unique(cl_label)))))
        count  = count + 1;
    end
end