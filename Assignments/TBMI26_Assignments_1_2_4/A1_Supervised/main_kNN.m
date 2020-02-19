%% This script will help you test out your kNN code

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 4; % Change this to load new data 
% X - Data samples
% D - Desired output from classifier for each sample
% L - Labels for each sample
[X, D, L] = loadDataSet( dataSetNr );

% You can plot and study dataset 1 to 3 by running:
% plotCase(X,D)

%% Select a subset of the training samples

numBins = 3;                    % Number of bins you want to devide your data into
numSamplesPerLabelPerBin = 100; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);

% Note: XBins, DBins, LBins will be cell arrays, to extract a single bin from them use e.g.
% XBin1 = XBins{1};
%
% Or use the combineBins helper function to combine several bins into one matrix (good for cross validataion)
% XBinComb = combineBins(XBins, [1,2,3]);

% Add your own code to setup data for training and test here
%indices = randperm(2000);
% XTrain = X(indices(1:1000), :);
% LTrain = L(indices(1:1000), :);
% XTest  = X(indices(1001:end), :);
% LTest  = L(indices(1001:end), :);

XTrain = combineBins(XBins, [1,2]);
LTrain = combineBins(LBins, [1,2]);
XTest = combineBins(XBins, (3));
LTest = combineBins(LBins, (3));

%% Use kNN to classify data
%  Note: you have to modify the kNN() function yourself.

% Set the number of neighbors


k = 4;

% Classify training data
LPredTrain = kNN(XTrain, k, XTrain, LTrain);
% Classify test data
LPredTest  = kNN(XTest , k, XTrain, LTrain);
%% Calculate The Confusion Matrix and the Accuracy
%  Note: you have to modify the calcConfusionMatrix() and calcAccuracy()
%  functions yourself.

% The confucionMatrix
cM = calcConfusionMatrix(LPredTest, LTest);

% The accuracy
acc = calcAccuracy(cM);

acc_list = [];
k_list = 1:30;
for k = k_list
    acc = 0;
    for i = 1:3
        x = mod(i+1, 3);
        y = mod(i+2, 3);
        if y == 0
            y = 3;
        end
        if x == 0
            x = 3;
        end
        XTrain_fold = combineBins(XBins, [i,x]);
        LTrain_fold = combineBins(LBins, [i,x]);
        XTest_fold = combineBins(XBins, (y));
        LTest_fold = combineBins(LBins, (y));
        LPredTest  = kNN(XTest_fold , k, XTrain_fold, LTrain_fold);
        cM = calcConfusionMatrix(LPredTest, LTest_fold);
        acc = acc + calcAccuracy(cM);
    end
    acc = acc / 3.0;
    acc_list(k) = acc;
end
figure(1);
plot(k_list, acc_list);
axis([1 30 0.98 1 ]);
title('CV: Dataset 4')
xlabel('K Neighbors')
ylabel('Accuracy')

best_ks = find(acc_list == max(acc_list));
k = best_ks(1)

LPredTrain = kNN(XTrain, k, XTrain, LTrain);
% Classify test data
LPredTest  = kNN(XTest , k, XTrain, LTrain);

cM = calcConfusionMatrix(LPredTest, LTest);

% The accuracy
acc = calcAccuracy(cM)

%% Plot classifications
%  Note: You should not have to modify this code

if dataSetNr < 4
    plotResultDots(XTrain, LTrain, LPredTrain, XTest, LTest, LPredTest, 'kNN', [], k);
else
    plotResultsOCR(XTest, LTest, LPredTest)
end
