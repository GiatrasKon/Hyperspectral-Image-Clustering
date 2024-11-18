clear
format compact
close all
DISPLAY_FIGURES = true;

% Load the Salinas hyperspectral image (HSI) and ground truth data
load('Salinas_cube.mat')  % Load the salinas_cube matrix
load('Salinas_gt.mat')   % Load the salinas_gt matrix

% Colormap of the data based on their class labels
if DISPLAY_FIGURES
    figure;
    colormap(jet);
    imagesc(salinas_gt), axis equal
    axis off
    title('Salinas Valley HSI Labeled Pixels Colormap')
    colorbar  
end

% ========================= Feeling the data =========================

% The Salinas HSI is now in a 220x120x204 cube
[p, n, l] = size(salinas_cube);  % p and n are spatial dimensions, l is spectral

% Reshape the HSI cube for clustering
X_total = reshape(salinas_cube, p*n, l);

% Adjust the ground truth labels
L = reshape(salinas_gt, p*n, 1);

% Filter out pixels with no class label (class label 0)
existed_L = (L > 0);  
X = X_total(existed_L, :);
[px, nx] = size(X);

% Update the ground truth to consider only existing labels
Ground_truth = L(existed_L);

% Define the new class labels for Salinas HSI
Labels = {1, 'Grapes'; 2, 'Broccoli'; 3, 'Fallow 1'; 4, 'Fallow 2'; 5, 'Fallow 3'; 6, 'Stubble'; 7, 'Celery'};
Labels_table = cell2table(Labels, 'VariableNames', {'Class_Number', 'Class_Label'});

% Plot PDF of each spectral band
if DISPLAY_FIGURES
    figure;
    for i = 1:204
        ksdensity(X(:,i))
        hold on
        grid on
    end
    xlabel('Wavelength(1e-6 meters)')
    title('PDF of each Spectral Band')
    hold off
end

% Find descriptive statistics about the dataset
max_X = max(max(X));
min_X = min(min(X));
mean_array = mean(X);
std_X = std(mean_array);

% ------------------------- Performing Mean Normalization -------------------------

% Calculate mean, max, and min for each column (spectral band)
mean_X_total = mean(X_total, 1);
max_X_total = max(X_total, [], 1);
min_X_total = min(X_total, [], 1);

% Normalize X_total using bsxfun for correct dimension matching
X_total_norm = bsxfun(@minus, X_total, mean_X_total);
X_total_norm = bsxfun(@rdivide, X_total_norm, (max_X_total - min_X_total));

% Extract the normalized data for existing labels
X = X_total_norm(existed_L, :);

% Plot PDF of each spectral band after mean normalization
if DISPLAY_FIGURES
    figure;
    for i = 1:204
        ksdensity(X(:,i))
        hold on
        grid on
    end
    xlabel('Wavelength(1e-6 meters)')
    title('PDF of each Spectral Band after Mean Normalization')
    hold off
end

% ------------------------- Performing PCA -------------------------

% Perform PCA on the normalized data using pca_fun
[eigenval, eigenvec, explain, Y, mean_vec] = pca_fun(X', 3);

% Extract the scores (the projections onto the principal components)
Z = Y';  % Transpose to match the pca output format

% Calculate the cumulative variance explained by the first three components
explained_variance = sum(explain(1:3));

% Placeholder for the first principal component with the original image dimensions
PC1_reshaped = zeros(p*n, 1);

% Fill in the values for the non-zero labeled pixels
PC1_reshaped(existed_L) = Z(:,1);

% Reshape the placeholder to the original spatial dimensions
PC1_reshaped = reshape(PC1_reshaped, p, n);

% Display the variance explained by the first three principal components
disp('Variance explained by the first three principal components:');
disp(array2table(explain(1:3)', 'VariableNames', {'First_PC', 'Second_PC', 'Third_PC'}));

% Plot the first principal component
if DISPLAY_FIGURES
    figure;
    imagesc(PC1_reshaped);
    colormap(jet);
    colorbar;
    title(sprintf('First Principal Component | %.2f%% of Total Variance', explain(1) * 100));
    axis off;
end

% Scatter plot of the first three principal components
if DISPLAY_FIGURES
    figure;
    scatter3(Z(:,1), Z(:,2), Z(:,3), 10, '.');  % The '10' is the size of the markers
    title('First Three Principal Components Scatter Plot');
    xlabel('First Principal Component');
    ylabel('Second Principal Component');
    zlabel('Third Principal Component');
    grid on;
    view([58 -6]);
end

% ========================= Identification of the Number of Physical Clusters =========================

% ------------------------- Elbow Method K-Means -------------------------

% Initialize arrays to store the sum of within-cluster distances
cost_stacked_total = [];
cost_stacked_PCA = [];

% Perform k-means clustering on the original dataset for a range of cluster numbers
for i = 1:40
    cost_total = [];
    for k = 2:10
        [idx, ~, sumd] = kmeans(X, k, 'Display', 'final', 'MaxIter', 1000, 'Replicates', 5);
        cost_total = cat(1, cost_total, sum(sumd));
    end
    cost_stacked_total = cat(2, cost_stacked_total, cost_total);
end

% Perform k-means clustering on the PCA reduced dataset for a range of cluster numbers
for i = 1:40
    cost_total = [];
    for k = 2:10
        [idx, ~, sumd] = kmeans(Z, k, 'Display', 'final', 'MaxIter', 1000, 'Replicates', 5);
        cost_total = cat(1, cost_total, sum(sumd));
    end
    cost_stacked_PCA = cat(2, cost_stacked_PCA, cost_total);
end

% Calculate the mean cost for each number of clusters
mean_cost_stacked_total = mean(cost_stacked_total, 2);
mean_cost_stacked_PCA = mean(cost_stacked_PCA, 2);

% Plot the results
if DISPLAY_FIGURES
    figure;
    subplot(1,2,1);
    plot(2:10, mean_cost_stacked_total);
    title('Elbow method | Original dataset')
    xlabel('Number of clusters')
    ylabel('Average cost function value')
    xticks(2:10)
    grid on;

    subplot(1,2,2);
    plot(2:10, mean_cost_stacked_PCA)
    title('Elbow method | PCA-reduced dataset')
    xlabel('Number of clusters')
    ylabel('Average cost function value')
    xticks(2:10)
    grid on;
end

% ------------------------- Silhouette Method K-Means -------------------------

% Perform silhouette analysis on the original normalized dataset
evaluation_total = evalclusters(X, 'kmeans', 'silhouette', 'KList', 1:10);

% Perform silhouette analysis on the PCA-reduced dataset
evaluation_PCA = evalclusters(Z, 'kmeans', 'silhouette', 'KList', 1:10);

% Plot the silhouette analysis results
if DISPLAY_FIGURES
    figure;
    
    % Silhouette analysis for the original normalized dataset
    subplot(1,2,1);
    plot(evaluation_total.CriterionValues);
    title('Silhouette method | Original dataset')
    xlabel('Number of clusters')
    ylabel('Average Silhouette Value')
    xticks(1:10);
    grid on;
    
    % Silhouette analysis for the PCA-reduced dataset
    subplot(1,2,2);
    plot(evaluation_PCA.CriterionValues);
    title('Silhouette method | PCA-reduced dataset')
    xlabel('Number of clusters')
    ylabel('Average Silhouette Value')
    xticks(1:10);
    grid on;
end

% ------------------------- Silhouette Method Hierarchical -------------------------

% Perform silhouette analysis using linkage on the original normalized dataset
evaluation_total = evalclusters(X, 'linkage', 'silhouette', 'KList', 1:10);

% Perform silhouette analysis using linkage on the PCA-reduced dataset
evaluation_PCA = evalclusters(Z, 'linkage', 'silhouette', 'KList', 1:10);

% Plot the silhouette analysis results
if DISPLAY_FIGURES
    figure;
    
    % Silhouette analysis for the original normalized dataset
    subplot(1,2,1);
    plot(evaluation_total.CriterionValues);
    title('Silhouette method | Original dataset')
    xlabel('Number of clusters')
    ylabel('Average Silhouette Value')
    xticks(1:10);
    grid on;
    
    % Silhouette analysis for the PCA-reduced dataset
    subplot(1,2,2);
    plot(evaluation_PCA.CriterionValues);
    title('Silhouette method | PCA-reduced dataset')
    xlabel('Number of clusters')
    ylabel('Average Silhouette Value')
    xticks(1:10);
    grid on;
end

% ========================= Execution of the Algorithms and Qualitative Evaluation =========================

% Plot ground truth labels in the space of the first two principal components
if DISPLAY_FIGURES
    figure;
    scatter(Z(:,1), Z(:,2), 8, Ground_truth, 'filled', 'MarkerEdgeAlpha', 0.9); % The '36' is the size of the markers
    title('Ground Truth labels in the space spanned by the first two PCs')
    xlabel('1st PC')
    ylabel('2nd PC')
    colormap('jet')
    colorbar
end

% Plot ground truth labels in the space of the first three principal components
if DISPLAY_FIGURES
    figure;
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Ground_truth, 'filled'); % The '36' is the size of the markers
    title('Ground Truth labels in the space spanned by the first three PCs')
    xlabel('1st PC')
    ylabel('2nd PC')
    zlabel('3rd PC')
    colormap('jet')
    colorbar
    view([58 -6])
end

% ------------------------- K-Means -------------------------

% Initialize variables to store the best silhouette scores and corresponding seeds
Best_silhouette_score_kmeans = zeros(1, 15);  % One entry for each seed
Best_seed = 0;
numClustersOptions = [6, 7];
Best_clustering_indices = cell(1, length(numClustersOptions));  % Store clustering results for each k

% Iterate over seed values
for seed = 1:15
    rng(seed, 'twister'); % Set the seed for reproducibility
    temp_clustering_indices = cell(1, length(numClustersOptions));  % Temporary storage for clustering indices
    silhouette_scores = zeros(1, length(numClustersOptions));  % Store silhouette scores for each k

    % Perform k-means clustering and silhouette analysis for each number of clusters
    for clusterOption = 1:length(numClustersOptions)
        currentNumClusters = numClustersOptions(clusterOption);
        idx = kmeans(Z, currentNumClusters, 'Distance', 'sqeuclidean');
        temp_clustering_indices{clusterOption} = idx;  % Store the clustering indices for this k
        s = silhouette(Z, idx, 'sqeuclidean');
        silhouette_scores(clusterOption) = mean(s);
    end

    % Compute the average silhouette score for the current seed
    average_silhouette_score = mean(silhouette_scores);

    % Store the seed, its score, and clustering indices if it is the best so far
    if average_silhouette_score > max(Best_silhouette_score_kmeans)
        Best_silhouette_score_kmeans = silhouette_scores;
        Best_seed = seed;
        Best_clustering_indices = temp_clustering_indices;  % Store the best clustering indices for each k
    end
end

% Display the best overall seed and the corresponding silhouette scores for each k
disp(['Best overall seed: ' num2str(Best_seed)]);
disp('Silhouette scores for the best seed:');
disp(array2table(Best_silhouette_score_kmeans, 'VariableNames', {'k6', 'k7'}));

% Reshape the first PC back to the original image dimensions for visualization
Z_total_cube = reshape(PC1_reshaped, p, n);

% Combine the best clustering results for both k=6 and k=7 into a matrix
Total_idx_kmeans = [Best_clustering_indices{1}, Best_clustering_indices{2}];

% Plot the Ground Truth, PC1, and clustering resufor lts
clust_eval(Total_idx_kmeans, 'k-means', Z_total_cube, salinas_gt, existed_L);

% Plot ground truth labels and clustering results for the first two principal components
if DISPLAY_FIGURES
    % Set the figure size
    figure('Position', [100, 100, 1200, 300]); % You can adjust the [left, bottom, width, height] as needed
    
    % Ground truth labels
    subplot('Position', [0.05, 0.1, 0.28, 0.8]); % [left, bottom, width, height] in normalized units
    scatter(Z(:,1), Z(:,2), 8, Ground_truth, 'filled', 'MarkerEdgeAlpha', 0.9);
    title('GT labels in the first two PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    colormap('jet');
    colorbar;
    axis square;
    
    % Clustering results for k=6
    subplot('Position', [0.37, 0.1, 0.28, 0.8]);
    scatter(Z(:,1), Z(:,2), 8, Total_idx_kmeans(:,1), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('k-means results (k=6) in the first two PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    colormap('jet');
    colorbar;
    axis square;
    
    % Clustering results for k=7
    subplot('Position', [0.69, 0.1, 0.28, 0.8]);
    scatter(Z(:,1), Z(:,2), 8, Total_idx_kmeans(:,2), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('k-means results (k=7) in the first two PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    colormap('jet');
    colorbar;
    axis square;
end

% Plot ground truth labels and clustering results for the first three principal components
if DISPLAY_FIGURES
    % Set the figure size
    figure('Position', [100, 100, 1200, 400]); % You may need to adjust the height to accommodate 3D plots
    
    % Ground truth labels in 3D
    subplot('Position', [0.05, 0.1, 0.28, 0.8]); % [left, bottom, width, height] in normalized units
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Ground_truth, 'filled', 'MarkerEdgeAlpha', 0.9);
    title('GT labels in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]); % Adjust the view angle for better visibility
    
    % Clustering results for k=6 in 3D
    subplot('Position', [0.37, 0.1, 0.28, 0.8]);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Total_idx_kmeans(:,1), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('k-means results (k=6) in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]); % Adjust the view angle for better visibility
    
    % Clustering results for k=7 in 3D
    subplot('Position', [0.69, 0.1, 0.28, 0.8]);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Total_idx_kmeans(:,2), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('k-means results (k=7) in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]); % Adjust the view angle for better visibility
end


% ------------------------- Fuzzy C-Means -------------------------

% Initialize variables to store the best silhouette scores and corresponding seeds for Fuzzy C-Means
Best_silhouette_score_FCM = zeros(1, 15);
Best_seed_FCM = 0;
numClustersOptions = [6, 7];
Best_clustering_indices_FCM = cell(1, length(numClustersOptions));

% Degree of fuzziness q = 2
options = [2; 100; 1e-5; 0]; % In order: [degree of fuzziness; iterations; error cutoff; verbosity]

% Iterate over seed values for Fuzzy C-Means
for seed = 1:15
    rng(seed, 'twister'); % Set the seed for reproducibility
    temp_clustering_indices_FCM = cell(1, length(numClustersOptions));
    silhouette_scores_FCM = zeros(1, length(numClustersOptions));

    % Perform Fuzzy C-Means clustering and silhouette analysis for each number of clusters
    for clusterOption = 1:length(numClustersOptions)
        currentNumClusters = numClustersOptions(clusterOption);
        [center, U] = fcm(Z, currentNumClusters, options);
        [~, maxU_index] = max(U, [], 1); % Find the index of the max membership value for each data point
        s = silhouette(Z, maxU_index', 'sqeuclidean');
        silhouette_scores_FCM(clusterOption) = mean(s);
        temp_clustering_indices_FCM{clusterOption} = maxU_index'; % Store the clustering indices for this k
    end

    % Compute the average silhouette score for the current seed
    average_silhouette_score_FCM = mean(silhouette_scores_FCM);

    % Store the seed, its score, and clustering indices if it is the best so far for Fuzzy C-Means
    if average_silhouette_score_FCM > max(Best_silhouette_score_FCM)
        Best_silhouette_score_FCM = silhouette_scores_FCM;
        Best_seed_FCM = seed;
        Best_clustering_indices_FCM = temp_clustering_indices_FCM; % Store the best clustering indices for each k
    end
end

% Display the best overall seed and the corresponding silhouette scores for Fuzzy C-Means
disp(['Best overall seed for Fuzzy C-Means: ' num2str(Best_seed_FCM)]);
disp('Silhouette scores for Fuzzy C-Means with the best seed:');
disp(array2table(Best_silhouette_score_FCM, 'VariableNames', {'k6', 'k7'}));

% Combine the best clustering results for both k=6 and k=7 into a matrix for Fuzzy C-Means
Total_idx_FCM = [Best_clustering_indices_FCM{1}, Best_clustering_indices_FCM{2}];

% Reshape the first PC back to the original image dimensions for visualization
Z_total_cube_FCM = reshape(PC1_reshaped, p, n);

% Plot the Ground Truth, PC1, and FCM clustering results
clust_eval(Total_idx_FCM, 'Fuzzy C-Means', Z_total_cube_FCM, salinas_gt, existed_L);

% Plot ground truth labels and FCM clustering results for the first two principal components
if DISPLAY_FIGURES
    % Set the figure size to accommodate three plots in a row
    figure('Position', [100, 100, 1500, 400]); % Adjust the [left, bottom, width, height] as needed
    
    % Ground truth labels
    subplot(1, 3, 1, 'Position', [0.05, 0.1, 0.27, 0.8]); % [left, bottom, width, height] in normalized units
    scatter(Z(:,1), Z(:,2), 10, Ground_truth, 'filled', 'MarkerEdgeAlpha', 0.9);
    title('Ground Truth labels in the first two PCs');
    xlabel('1st PC');
    ylabel('2nd PC');
    axis square;
    set(gca, 'Colormap', jet);
    colorbar;
    
    % FCM clustering results for k=6
    subplot(1, 3, 2, 'Position', [0.37, 0.1, 0.27, 0.8]);
    scatter(Z(:,1), Z(:,2), 10, Total_idx_FCM(:,1), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('FCM results (k=6) in the first two PCs');
    xlabel('1st PC');
    ylabel('2nd PC');
    axis square;
    set(gca, 'Colormap', jet);
    colorbar;
    
    % FCM clustering results for k=7
    subplot(1, 3, 3, 'Position', [0.69, 0.1, 0.27, 0.8]);
    scatter(Z(:,1), Z(:,2), 10, Total_idx_FCM(:,2), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('FCM results (k=7) in the first two PCs');
    xlabel('1st PC');
    ylabel('2nd PC');
    axis square;
    set(gca, 'Colormap', jet);
    colorbar;
end

% Plot ground truth labels and FCM clustering results for the first three principal components
if DISPLAY_FIGURES
    % Set the figure size
    figure('Position', [100, 100, 1200, 400]); % You may need to adjust the height to accommodate 3D plots
    
    % Ground truth labels in 3D
    subplot('Position', [0.05, 0.1, 0.28, 0.8]); % [left, bottom, width, height] in normalized units
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Ground_truth, 'filled');
    title('GT labels in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]); % Adjust the view angle for better visibility
    
    % FCM clustering results for k=6 in 3D
    subplot('Position', [0.37, 0.1, 0.28, 0.8]);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Total_idx_FCM(:,1), 'filled');
    title('FCM results (k=6) in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]); % Adjust the view angle for better visibility
    
    % FCM clustering results for k=7 in 3D
    subplot('Position', [0.69, 0.1, 0.28, 0.8]);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Total_idx_FCM(:,2), 'filled');
    title('FCM results (k=7) in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]); % Adjust the view angle for better visibility
end


% ------------------------- Possibilistic C-Means -------------------------

% Initialize variables for PCM
Best_silhouette_score_PCM = zeros(1, 15);
Best_seed_PCM = 0;
numClustersOptions = [6, 7];
Best_clustering_indices_PCM = cell(1, length(numClustersOptions));

% Transpose Z for compatibility with possibi function
Z_transposed = Z';

% Iterate over seed values for Possibilistic C-Means
for seed = 1:15
    rng(seed, 'twister'); % Set the seed for reproducibility
    temp_clustering_indices_PCM = cell(1, length(numClustersOptions));
    silhouette_scores_PCM = zeros(1, length(numClustersOptions));

    % Calculate eta for PCM
    mean_Z = mean(Z, 2);
    diff_Z = bsxfun(@minus, Z, mean_Z);
    eta = mean(sum(diff_Z .^ 2, 2));

    % Perform Possibilistic C-Means clustering and silhouette analysis for each number of clusters
    for clusterOption = 1:length(numClustersOptions)
        currentNumClusters = numClustersOptions(clusterOption);
        eta_m = ones(1, currentNumClusters) * eta;
        
        % Perform PCM clustering using the 'possibi' function with transposed Z
        [U, ~] = possibi(Z_transposed, currentNumClusters, eta_m, 2, seed, 3, 0.001); % Adjust the 'possibi' function parameters as needed
        [~, maxU_index] = max(U, [], 2);
        
        % Calculate the silhouette score
        s = silhouette(Z, maxU_index, 'sqeuclidean');
        silhouette_scores_PCM(clusterOption) = mean(s);
        temp_clustering_indices_PCM{clusterOption} = maxU_index;
    end

    % Compute the average silhouette score for the current seed
    average_silhouette_score_PCM = mean(silhouette_scores_PCM);

    % Store the seed, its score, and clustering indices if it is the best so far for Possibilistic C-Means
    if average_silhouette_score_PCM > max(Best_silhouette_score_PCM)
        Best_silhouette_score_PCM = silhouette_scores_PCM;
        Best_seed_PCM = seed;
        Best_clustering_indices_PCM = temp_clustering_indices_PCM; % Store the best clustering indices for each k
    end
end

% Display the best overall seed and the corresponding silhouette scores for Possibilistic C-Means
disp(['Best overall seed for Possibilistic C-Means: ' num2str(Best_seed_PCM)]);
disp('Silhouette scores for Possibilistic C-Means with the best seed:');
disp(array2table(Best_silhouette_score_PCM, 'VariableNames', {'k6', 'k7'}));

% Combine the best clustering results for both k=6 and k=7 into a matrix for Possibilistic C-Means
Total_idx_PCM = [Best_clustering_indices_PCM{1}, Best_clustering_indices_PCM{2}];

% Reshape the first PC back to the original image dimensions for visualization
Z_total_cube_PCM = reshape(PC1_reshaped, p, n);

% Plot the Ground Truth, PC1, and PCM clustering results
clust_eval(Total_idx_PCM, 'Possibilistic C-Means', Z_total_cube_PCM, salinas_gt, existed_L);

if DISPLAY_FIGURES
    % Set the figure size to accommodate three plots in a row
    figure('Position', [100, 100, 1500, 400]); % Adjust as needed
    
    % Ground truth labels
    subplot(1, 3, 1, 'Position', [0.05, 0.1, 0.27, 0.8]);
    scatter(Z(:,1), Z(:,2), 10, Ground_truth, 'filled', 'MarkerEdgeAlpha', 0.9);
    title('Ground Truth labels in the first two PCs');
    xlabel('1st PC');
    ylabel('2nd PC');
    axis square;
    colormap('jet');
    colorbar;
    
    % PCM clustering results for k=6
    subplot(1, 3, 2, 'Position', [0.37, 0.1, 0.27, 0.8]);
    scatter(Z(:,1), Z(:,2), 10, Total_idx_PCM(:,1), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('PCM results (k=6) in the first two PCs');
    xlabel('1st PC');
    ylabel('2nd PC');
    axis square;
    colormap('jet');
    colorbar;
    
    % PCM clustering results for k=7
    subplot(1, 3, 3, 'Position', [0.69, 0.1, 0.27, 0.8]);
    scatter(Z(:,1), Z(:,2), 10, Total_idx_PCM(:,2), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('PCM results (k=7) in the first two PCs');
    xlabel('1st PC');
    ylabel('2nd PC');
    axis square;
    colormap('jet');
    colorbar;
end

if DISPLAY_FIGURES
    % Set the figure size for 3D plots
    figure('Position', [100, 100, 1200, 400]); % Adjust as needed
    
    % Ground truth labels in 3D
    subplot('Position', [0.05, 0.1, 0.28, 0.8]);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Ground_truth, 'filled');
    title('GT labels in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]);
    
    % PCM clustering results for k=6 in 3D
    subplot('Position', [0.37, 0.1, 0.28, 0.8]);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Total_idx_PCM(:,1), 'filled');
    title('PCM results (k=6) in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]);
    
    % PCM clustering results for k=7 in 3D
    subplot('Position', [0.69, 0.1, 0.28, 0.8]);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Total_idx_PCM(:,2), 'filled');
    title('PCM results (k=7) in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]);
end


% ------------------------- Probabilistic C-Means -------------------------

% Initialize variables for PBCM
Best_negLogLikelihood_PBCM = inf(1, 15);  % Store the best negative log-likelihood for each seed
Best_seeds_PBCM = zeros(1, 15);  % Store the best seeds
numClustersOptions = [6, 7];  % Cluster options
Best_clustering_indices_PBCM = cell(length(numClustersOptions), 15);  % Store clustering results for each k and each seed

% Iterate over seeds and cluster options
for seed = 1:15
    rng(seed, 'twister'); % Set the seed for reproducibility
    
    for clusterOption = 1:length(numClustersOptions)
        currentNumClusters = numClustersOptions(clusterOption);
        GMModel = fitgmdist(Z, currentNumClusters, 'CovarianceType', 'full', 'Replicates', 1);

        % Cluster assignment and model evaluation
        [~, nLogL] = cluster(GMModel, Z);

        % Update the best model based on negative log-likelihood
        if nLogL < Best_negLogLikelihood_PBCM(seed)
            Best_negLogLikelihood_PBCM(seed) = nLogL;
            Best_seeds_PBCM(seed) = seed;
            Best_clustering_indices_PBCM{clusterOption, seed} = cluster(GMModel, Z);
        end
    end
end

% Find the overall best seed based on the negative log-likelihood
[~, bestSeedIndex] = min(Best_negLogLikelihood_PBCM);
bestSeed = Best_seeds_PBCM(bestSeedIndex);

% Display the best overall seed and the corresponding negative log-likelihoods
disp(['Best overall seed: ' num2str(bestSeed)]);
for clusterOption = 1:length(numClustersOptions)
    disp(['Negative Log Likelihood for ' num2str(numClustersOptions(clusterOption)) ' clusters: ' num2str(Best_negLogLikelihood_PBCM(bestSeedIndex))]);
end

% Combine the best clustering results for both k=6 and k=7 into a matrix for PBCM
Total_idx_PBCM = [Best_clustering_indices_PBCM{1}, Best_clustering_indices_PBCM{2}];

% Reshape the first principal component back to the original image dimensions for visualization
Z_total_cube_PBCM = reshape(PC1_reshaped, p, n);

% Plot the Ground Truth, first principal component, and PBCM clustering results
clust_eval(Total_idx_PBCM, 'Probabilistic C-Means', Z_total_cube_PBCM, salinas_gt, existed_L);

% Plot ground truth labels and PBCM clustering results for the first two principal components
if DISPLAY_FIGURES
    % Set the figure size
    figure('Position', [100, 100, 1200, 300]);
    
    % Ground truth labels
    subplot('Position', [0.05, 0.1, 0.28, 0.8]);
    scatter(Z(:,1), Z(:,2), 8, Ground_truth, 'filled', 'MarkerEdgeAlpha', 0.9);
    title('GT labels in the first two PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    colormap('jet');
    colorbar;
    axis square;
    
    % PBCM clustering results for k=6
    subplot('Position', [0.37, 0.1, 0.28, 0.8]);
    scatter(Z(:,1), Z(:,2), 8, Total_idx_PBCM(:,1), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('PBCM results (k=6) in the first two PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    colormap('jet');
    colorbar;
    axis square;
    
    % PBCM clustering results for k=7
    subplot('Position', [0.69, 0.1, 0.28, 0.8]);
    scatter(Z(:,1), Z(:,2), 8, Total_idx_PBCM(:,2), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('PBCM results (k=7) in the first two PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    colormap('jet');
    colorbar;
    axis square;
end

% Plot ground truth labels and PBCM clustering results for the first three principal components
if DISPLAY_FIGURES
    % Set the figure size for 3D plots
    figure('Position', [100, 100, 1200, 400]); % Adjust as needed
    
    % Ground truth labels in 3D
    subplot('Position', [0.05, 0.1, 0.28, 0.8]);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Ground_truth, 'filled', 'MarkerEdgeAlpha', 0.9);
    title('GT labels in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]); % Adjust the view angle
    
    % PBCM clustering results for k=6 in 3D
    subplot('Position', [0.37, 0.1, 0.28, 0.8]);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Total_idx_PBCM(:,1), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('PBCM results (k=6) in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]); % Adjust the view angle
    
    % PBCM clustering results for k=7 in 3D
    subplot('Position', [0.69, 0.1, 0.28, 0.8]);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Total_idx_PBCM(:,2), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('PBCM results (k=7) in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]); % Adjust the view angle
end


% ------------------------- Complete-Link Algorithm -------------------------

% Perform Complete Link Hierarchical Clustering for specified cluster numbers
Total_idx_CL = [];
for i = [6, 7]
    B = linkage(Z, 'complete', 'euclidean');
    idx = cluster(B, 'maxclust', i);
    Total_idx_CL = cat(2, Total_idx_CL, idx);
end

% Reshape the first PC back to the original image dimensions for visualization
Z_total_cube_CL = reshape(PC1_reshaped, p, n);

% Plot the Ground Truth, PC1, and Complete Link clustering results
clust_eval(Total_idx_CL, 'Complete Link', Z_total_cube_CL, salinas_gt, existed_L);

if DISPLAY_FIGURES
    % Set the figure size for 2D plots
    figure('Position', [100, 100, 1200, 300]);
    
    % Ground truth labels for the first two principal components
    subplot('Position', [0.05, 0.1, 0.28, 0.8]);
    scatter(Z(:,1), Z(:,2), 8, Ground_truth, 'filled', 'MarkerEdgeAlpha', 0.9);
    title('Ground Truth labels in the first two PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    colormap('jet');
    colorbar;
    axis square;

    % Complete Link clustering results for k=6 in the first two principal components
    subplot('Position', [0.37, 0.1, 0.28, 0.8]);
    scatter(Z(:,1), Z(:,2), 8, Total_idx_CL(:,1), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('Complete Link results (k=6) in the first two PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    colormap('jet');
    colorbar;
    axis square;

    % Complete Link clustering results for k=7 in the first two principal components
    subplot('Position', [0.69, 0.1, 0.28, 0.8]);
    scatter(Z(:,1), Z(:,2), 8, Total_idx_CL(:,2), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('Complete Link results (k=7) in the first two PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    colormap('jet');
    colorbar;
    axis square;
end

if DISPLAY_FIGURES
    % Set the figure size for 3D plots
    figure('Position', [100, 100, 1200, 400]);
    
    % Ground truth labels for the first three principal components
    subplot('Position', [0.05, 0.1, 0.28, 0.8]);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Ground_truth, 'filled');
    title('Ground Truth labels in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]);

    % Complete Link clustering results for k=6 in the first three principal components
    subplot('Position', [0.37, 0.1, 0.28, 0.8]);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Total_idx_CL(:,1), 'filled');
    title('Complete Link results (k=6) in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]);

    % Complete Link clustering results for k=7 in the first three principal components
    subplot('Position', [0.69, 0.1, 0.28, 0.8]);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Total_idx_CL(:,2), 'filled');
    title('Complete Link results (k=7) in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]);
end



% ------------------------- WPGMC -------------------------

% Perform WPGMC Hierarchical Clustering for specified cluster numbers
Total_idx_WPGMC = [];
for i = [6, 7]
    % Using 'median' method in linkage for WPGMC
    B = linkage(Z, 'median', 'euclidean');
    idx = cluster(B, 'maxclust', i);
    Total_idx_WPGMC = cat(2, Total_idx_WPGMC, idx);
end

% Reshape the first PC back to the original image dimensions for visualization
Z_total_cube_WPGMC = reshape(PC1_reshaped, p, n);

% Plot the Ground Truth, PC1, and WPGMC clustering results
clust_eval(Total_idx_WPGMC, 'WPGMC Algorithm', Z_total_cube_WPGMC, salinas_gt, existed_L);

if DISPLAY_FIGURES
    % Set the figure size for 2D plots
    figure('Position', [100, 100, 1200, 300]);
    
    % Ground truth labels for the first two principal components
    subplot(1, 3, 1);
    scatter(Z(:,1), Z(:,2), 8, Ground_truth, 'filled', 'MarkerEdgeAlpha', 0.9);
    title('Ground Truth labels in the first two PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    colormap('jet');
    colorbar;
    axis square;

    % WPGMC clustering results for k=6 in the first two principal components
    subplot(1, 3, 2);
    scatter(Z(:,1), Z(:,2), 8, Total_idx_WPGMC(:,1), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('WPGMC results (k=6) in the first two PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    colormap('jet');
    colorbar;
    axis square;

    % WPGMC clustering results for k=7 in the first two principal components
    subplot(1, 3, 3);
    scatter(Z(:,1), Z(:,2), 8, Total_idx_WPGMC(:,2), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('WPGMC results (k=7) in the first two PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    colormap('jet');
    colorbar;
    axis square;
end

if DISPLAY_FIGURES
    % Set the figure size for 3D plots
    figure('Position', [100, 100, 1200, 400]);
    
    % Ground truth labels for the first three principal components
    subplot(1, 3, 1);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Ground_truth, 'filled');
    title('Ground Truth labels in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]);

    % WPGMC clustering results for k=6 in the first three principal components
    subplot(1, 3, 2);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Total_idx_WPGMC(:,1), 'filled');
    title('WPGMC results (k=6) in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]);

    % WPGMC clustering results for k=7 in the first three principal components
    subplot(1, 3, 3);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Total_idx_WPGMC(:,2), 'filled');
    title('WPGMC results (k=7) in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]);
end


% ------------------------- Ward's Algorithm -------------------------

% Perform Ward's Hierarchical Clustering for specified cluster numbers
Total_idx_Ward = [];
for i = [6, 7]
    B = linkage(Z, 'ward', 'euclidean');
    idx = cluster(B, 'maxclust', i);
    Total_idx_Ward = cat(2, Total_idx_Ward, idx);
end

% Reshape the first PC back to the original image dimensions for visualization
Z_total_cube_Ward = reshape(PC1_reshaped, p, n);

% Plot the Ground Truth, PC1, and Ward's clustering results
clust_eval(Total_idx_Ward, 'Ward’s Algorithm', Z_total_cube_Ward, salinas_gt, existed_L);

% Plot ground truth labels and Ward's clustering results for the first two principal components
if DISPLAY_FIGURES
    % Set the figure size for 2D plots
    figure('Position', [100, 100, 1200, 300]);
    
    % Ground truth labels for the first two principal components
    subplot('Position', [0.05, 0.1, 0.28, 0.8]);
    scatter(Z(:,1), Z(:,2), 8, Ground_truth, 'filled', 'MarkerEdgeAlpha', 0.9);
    title('Ground Truth labels in the first two PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    colormap('jet');
    colorbar;
    axis square;

    % Ward's clustering results for k=6 in the first two principal components
    subplot('Position', [0.37, 0.1, 0.28, 0.8]);
    scatter(Z(:,1), Z(:,2), 8, Total_idx_Ward(:,1), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('Ward’s Algorithm results (k=6) in the first two PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    colormap('jet');
    colorbar;
    axis square;

    % Ward's clustering results for k=7 in the first two principal components
    subplot('Position', [0.69, 0.1, 0.28, 0.8]);
    scatter(Z(:,1), Z(:,2), 8, Total_idx_Ward(:,2), 'filled', 'MarkerEdgeAlpha', 0.9);
    title('Ward’s Algorithm results (k=7) in the first two PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    colormap('jet');
    colorbar;
    axis square;
end

% Plot ground truth labels and Ward's clustering results for the first three principal components
if DISPLAY_FIGURES
    % Set the figure size for 3D plots
    figure('Position', [100, 100, 1200, 400]);
    
    % Ground truth labels for the first three principal components
    subplot('Position', [0.05, 0.1, 0.28, 0.8]);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Ground_truth, 'filled');
    title('Ground Truth labels in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]);

    % Ward's clustering results for k=6 in the first three principal components
    subplot('Position', [0.37, 0.1, 0.28, 0.8]);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Total_idx_Ward(:,1), 'filled');
    title('Ward’s Algorithm results (k=6) in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]);

    % Ward's clustering results for k=7 in the first three principal components
    subplot('Position', [0.69, 0.1, 0.28, 0.8]);
    scatter3(Z(:,1), Z(:,2), Z(:,3), 8, Total_idx_Ward(:,2), 'filled');
    title('Ward’s Algorithm results (k=7) in the first three PCs space');
    xlabel('1st PC');
    ylabel('2nd PC');
    zlabel('3rd PC');
    colormap('jet');
    colorbar;
    axis square;
    view([30 20]);
end

% ========================= Quantitative Evaluation of the Results =========================

% Evaluate k-means clustering results using Rand Index for k=6
randIndex_k6 = rand_index(Total_idx_kmeans(:,1), Ground_truth);
disp(['Rand Index for k-means with k=6: ', num2str(randIndex_k6)]);

% Evaluate k-means clustering results using Rand Index for k=7
randIndex_k7 = rand_index(Total_idx_kmeans(:,2), Ground_truth);
disp(['Rand Index for k-means with k=7: ', num2str(randIndex_k7)]);

% Evaluate FCM clustering results using Rand Index for k=6
randIndex_FCM_k6 = rand_index(Total_idx_FCM(:,1), Ground_truth);
disp(['Rand Index for Fuzzy C-Means with k=6: ', num2str(randIndex_FCM_k6)]);

% Evaluate FCM clustering results using Rand Index for k=7
randIndex_FCM_k7 = rand_index(Total_idx_FCM(:,2), Ground_truth);
disp(['Rand Index for Fuzzy C-Means with k=7: ', num2str(randIndex_FCM_k7)]);

% Evaluate PCM clustering results using Rand Index for k=6
randIndex_PCM_k6 = rand_index(Total_idx_PCM(:,1), Ground_truth);
disp(['Rand Index for Possibilistic C-Means with k=6: ', num2str(randIndex_PCM_k6)]);

% Evaluate PCM clustering results using Rand Index for k=7
randIndex_PCM_k7 = rand_index(Total_idx_PCM(:,2), Ground_truth);
disp(['Rand Index for Possibilistic C-Means with k=7: ', num2str(randIndex_PCM_k7)]);

% Evaluate PBCM clustering results using Rand Index for k=6
randIndex_PBCM_k6 = rand_index(Total_idx_PBCM(:,1), Ground_truth);
disp(['Rand Index for Probabilistic C-Means with k=6: ', num2str(randIndex_PBCM_k6)]);

% Evaluate PBCM clustering results using Rand Index for k=7
randIndex_PBCM_k7 = rand_index(Total_idx_PBCM(:,2), Ground_truth);
disp(['Rand Index for Probabilistic C-Means with k=7: ', num2str(randIndex_PBCM_k7)]);

% Evaluate Complete-Link clustering results using Rand Index for k=6
randIndex_CL_k6 = rand_index(Total_idx_CL(:,1), Ground_truth);
disp(['Rand Index for Complete-Link with k=6: ', num2str(randIndex_CL_k6)]);

% Evaluate Complete-Link clustering results using Rand Index for k=7
randIndex_CL_k7 = rand_index(Total_idx_CL(:,2), Ground_truth);
disp(['Rand Index for Complete-Link with k=7: ', num2str(randIndex_CL_k7)]);

% Evaluate WPGMC clustering results using Rand Index for k=6
randIndex_WPGMC_k6 = rand_index(Total_idx_WPGMC(:,1), Ground_truth);
disp(['Rand Index for WPGMC with k=6: ', num2str(randIndex_WPGMC_k6)]);

% Evaluate WPGMC clustering results using Rand Index for k=7
randIndex_WPGMC_k7 = rand_index(Total_idx_WPGMC(:,2), Ground_truth);
disp(['Rand Index for WPGMC with k=7: ', num2str(randIndex_WPGMC_k7)]);

% Evaluate Ward's clustering results using Rand Index for k=6
randIndex_Ward_k6 = rand_index(Total_idx_Ward(:,1), Ground_truth);
disp(['Rand Index for Ward Algorithm with k=6: ', num2str(randIndex_Ward_k6)]);

% Evaluate Ward's clustering results using Rand Index for k=7
randIndex_Ward_k7 = rand_index(Total_idx_Ward(:,2), Ground_truth);
disp(['Rand Index for Ward Algorithm with k=7: ', num2str(randIndex_Ward_k7)]);

% Data for Rand Index values
algorithms = {'K-Means', 'FCM', 'PCM', 'PBCM', 'CL', 'WPGMC', 'Ward'};
randIndex_6Clusters = [randIndex_k6, randIndex_FCM_k6, randIndex_PCM_k6, randIndex_PBCM_k6, randIndex_CL_k6, randIndex_WPGMC_k6, randIndex_Ward_k6];
randIndex_7Clusters = [randIndex_k7, randIndex_FCM_k7, randIndex_PCM_k7, randIndex_PBCM_k7, randIndex_CL_k7, randIndex_WPGMC_k7, randIndex_Ward_k7];

% Bar plot for 6 clusters
figure;
bar(randIndex_6Clusters);
set(gca, 'xticklabel', algorithms);
ylabel('Rand Index');
title('Rand Index for Various Clustering Algorithms with 6 Clusters');
ylim([0.8 1]); % Assuming the Rand Index values are between 0.8 and 1

% Adding the values on top of the bars
for i = 1:length(randIndex_6Clusters)
    text(i, randIndex_6Clusters(i), num2str(randIndex_6Clusters(i), '%0.2f'), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
end

% Bar plot for 7 clusters
figure;
bar(randIndex_7Clusters);
set(gca, 'xticklabel', algorithms);
ylabel('Rand Index');
title('Rand Index for Various Clustering Algorithms with 7 Clusters');
ylim([0.8 1]); % Adjust the limits based on your data range

% Adding the values on top of the bars
for i = 1:length(randIndex_7Clusters)
    text(i, randIndex_7Clusters(i), num2str(randIndex_7Clusters(i), '%0.2f'), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
end


% ========================= FUNCTIONS =========================

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

function RI = rand_index(clustering1, clustering2)
    %RAND_INDEX Computes the Rand index to evaluate clustering performance
    %   RI = rand_index(clustering1, clustering2) returns the Rand index, which is
    %   a measure of the similarity between two data clusterings.

    n = length(clustering1);  % Number of elements

    % Initialize counts
    a = 0; % Pairs clustered together in both clusterings
    b = 0; % Pairs clustered separately in both clusterings

    % Iterate over all pairs of elements
    for i = 1:n-1
        for j = i+1:n
            % Check if the pair is clustered similarly in both clusterings
            if (clustering1(i) == clustering1(j) && clustering2(i) == clustering2(j)) || (clustering1(i) ~= clustering1(j) && clustering2(i) ~= clustering2(j))
                a = a + 1;
            else
                b = b + 1;
            end
        end
    end

    % Calculate Rand index
    RI = a / (a + b);
end