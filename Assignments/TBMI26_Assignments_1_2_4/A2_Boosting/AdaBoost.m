%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 100;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 1000;
% Number of weak classifiers
nbrWeakClassifiers = 70;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError

D = ones(1,size(xTrain,2))*1/size(xTrain,2);
%H = zeros(1,size(xTrain,2));

for i = 1:nbrWeakClassifiers
    bestPreds = 0;
    bestP = 0;
    bestThreshold = 0;
    bestErr = inf;
    bestfeatureNr = 0;
    for featureNr = 1:size(xTrain, 1)
         sortedFeatures = sort(xTrain(featureNr,:));
         for j = 1:size(sortedFeatures, 2)-1
            p = 1;
            thresh = (sortedFeatures(j) + sortedFeatures(j + 1)) / 2;
            preds = WeakClassifier(thresh,p,xTrain(featureNr,:));
            Err = WeakClassifierError(preds,D,yTrain);
            if (Err > 0.5)
                p = -1;
                Err = 1 - Err;
            end
            if (bestErr > Err)
                bestErr = Err;
                bestPreds = preds*p;
                bestfeatureNr = featureNr;
                bestThreshold = thresh;
                bestP = p;
            end
        end
    end
    alpha = .5 * log((1-bestErr) / bestErr);
    D = D .* exp(-alpha * yTrain .* bestPreds) / sum(D);
    thresholds(i) = bestThreshold;
    Ps(i) = bestP;
    features(i) = bestfeatureNr;
    alphas(i) = alpha;
end
%% Evaluate your strong classifier here

temp = zeros(1, size(xTest, 2));
for k = 1 : nbrWeakClassifiers
    preds_test = WeakClassifier(thresholds(k),Ps(k),xTest(features(k),:));
    temp = temp + preds_test * alphas(k);
    test_acc(k) = sum(sign(temp) == yTest) / size(yTest, 2);
end

temp = zeros(1, size(xTrain, 2));
for k = 1 : nbrWeakClassifiers
    preds_train = WeakClassifier(thresholds(k),Ps(k),xTrain(features(k),:));
    temp = temp + preds_train * alphas(k);
    train_acc(k) = sum(sign(temp) == yTrain) / size(yTrain, 2);
end

%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.
final_train_acc = train_acc(:,length(train_acc))

final_test_acc = test_acc(:,length(test_acc))

%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.
figure(4);
%test_error = 1-test_acc;
plot(1:nbrWeakClassifiers,test_acc, 'r-');
hold on;
plot(1:nbrWeakClassifiers,train_acc, 'b-');
title("accuracy on test and training set depending on nr of weak classifiers")
hold off
legend('test','train')


%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.
missclass_faces = find(yTest == 1 & yTest ~= preds_test);
missclass_nonfaces = find(yTest == -1 & yTest ~= preds_test);
figure(6);
colormap gray;
for k=1:20
    subplot(5,5,k), imagesc(testImages(:,:,missclass_faces(k)));
    axis image;
    axis off;
end

figure(7);
colormap gray;
for k=1:20
    subplot(5,5,k), imagesc(testImages(:,:,missclass_nonfaces(k)));
    axis image;
    axis off;
end

%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.
figure(8);
colormap gray;
indx = 1;
for k = features(1:25)
    subplot(5,5,indx),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    indx = indx + 1;
    axis image;
    axis off;
end
