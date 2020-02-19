function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)
classes = unique(LTrain);
NClasses = length(classes);
distanceMatrix = pdist2(XTrain, X);
[dist, idx] = sort(distanceMatrix);
idx = idx(1:k, :);
if k == 1
    res = LTrain(idx);
else
    res = mode(LTrain(idx));
end

LPred = transpose(res);
    



%for x in X:
%    neighbors = get_k_neighbors(x)
%    prediction = most_frequent(neighbors)

% Add your own code here
%LPred  = zeros(size(X,1),1);

end

