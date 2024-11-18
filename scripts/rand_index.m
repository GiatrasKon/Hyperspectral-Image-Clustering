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