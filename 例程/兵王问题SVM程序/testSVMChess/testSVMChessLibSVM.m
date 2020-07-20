% fid  =  fopen('krkopt.data'); % Read data
% c = fread(fid, 3); % Useless data
% 
% vec = zeros(6,1); % Store position
% xapp = []; % Data sets
% yapp = []; % Labels
% while ~feof(fid)
%     string = [];
%     c = fread(fid, 1); % Read 1 byte
%     while c ~= 13 % c != '\x0d'
%         string = [string, c];
%         c = fread(fid,1); % Read 1 byte
%     end
%     fread(fid, 1); % Read '\x0a'
%     if length(string) > 10
%         vec(1) = string(1) - 96; % Letter 1
%         vec(2) = string(3) - 48; % Number 1
%         vec(3) = string(5) - 96; % Letter 2
%         vec(4) = string(7) - 48; % Number 2
%         vec(5) = string(9) - 96; % Letter 3
%         vec(6) = string(11) - 48; % Number 3
%         xapp = [xapp, vec];
%         if string(13) == 100 % 'draw'
%             yapp = [yapp, 1];
%         else
%             yapp = [yapp, -1];
%         end
%     end
% end
% fclose(fid);
% 
% [N, M] = size(xapp);
% p = randperm(M); % 直接打乱了训练样本
% numberOfSamplesForTraining = 5000; % Number of training samples
% xTraining = [];
% yTraining = [];
% for i = 1 : numberOfSamplesForTraining
%     xTraining = [xTraining, xapp(:, p(i))]; % Get training data sets
%     yTraining = [yTraining, yapp(p(i))]; % Get training labels
% end
% xTraining = xTraining'; % Transpose
% yTraining = yTraining'; % Transpose
% 
% xTesting = [];
% yTesting = [];
% for i = numberOfSamplesForTraining + 1 : M
%     xTesting = [xTesting, xapp(:, p(i))]; % Get testing data sets
%     yTesting = [yTesting, yapp(p(i))]; % Get testing labels
% end
% xTesting = xTesting'; % Transpose
% yTesting = yTesting'; % Transpose
% 
% [numVec, numDim] = size(xTraining);
% avgX = mean(xTraining); % Return the average of training data sets
% stdX = std(xTraining); % Return the standard deviation of training data sets
% disp(avgX);
% disp(stdX);
% for i = 1 : numVec
%     xTraining(i, :) = (xTraining(i, :) - avgX) ./ stdX; % Normal divide
% end
% 
% [numVec, numDim] = size(xTesting);
% for i = 1 : numVec
%     xTesting(i, :) = (xTesting(i, :) - avgX) ./ stdX; % Normal divide
% end
% 
% % Search for the optimal C and gamma, K(x1,x2) = exp{-||x1-x2||^2/gamma}
% % Firstly, search C and gamma in a crude scale
% CScale = [-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15];
% gammaScale = [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3];
% C = 2 .^ CScale;
% gamma = 2 .^ gammaScale;
% maxRecognitionRate = 0;
% for i = 1 : length(C)
%     for j = 1 : length(gamma)
%         % 5-fold cross validation
%         cmd = ['-t 2 -c ', num2str(C(i)), ' -g ', num2str(gamma(j)), ' -v 5 -q'];
%         recognitionRate = libsvmtrain(yTraining, xTraining, cmd);
%         if recognitionRate > maxRecognitionRate
%             maxRecognitionRate = recognitionRate;
%             maxCIndex = i;
%             maxGammaIndex = j;
%         end
%     end
% end
% 
% % Then search for optimal C and gamma in a refined scale. 
% n = 10;
% minCScale = 0.5 * (CScale(max(1, maxCIndex - 1)) + CScale(maxCIndex));
% maxCScale = 0.5 * (CScale(min(length(CScale), maxCIndex + 1)) + CScale(maxCIndex));
% newCScale = [minCScale : (maxCScale - minCScale) / n : maxCScale];
% 
% minGammaScale = 0.5 * (gammaScale(max(1, maxGammaIndex - 1)) + gammaScale(maxGammaIndex));
% maxGammaScale = 0.5 * (gammaScale(min(length(gammaScale), maxGammaIndex + 1)) + gammaScale(maxGammaIndex));
% newGammaScale = [minGammaScale : (maxGammaScale - minGammaScale) / n : maxGammaScale];
% newC = 2 .^ newCScale;
% newGamma = 2 .^ newGammaScale;
% maxRecognitionRate = 0;
% for i = 1 : length(newC)
%     for j = 1 : length(newGamma)
%         % 5-fold cross validation
%         cmd = ['-t 2 -c ', num2str(newC(i)), ' -g ', num2str(newGamma(j)), ' -v 5 -q'];
%         recognitionRate = libsvmtrain(yTraining, xTraining, cmd);
%         if recognitionRate > maxRecognitionRate % Iterate and find the maximum rate
%             maxRecognitionRate = recognitionRate
%             maxC = newC(i);
%             maxGamma = newGamma(j);
%         end
%     end
% end
% 
% % Train the SVM model by the optimal C and gamma.
% cmd = ['-t 2 -c ', num2str(maxC), ' -g ', num2str(maxGamma)];
% model = libsvmtrain(yTraining, xTraining, cmd);
% save model.mat model;
% 
% % Test the model on the remaining testing data and obtain the recognition rate.
% load model.mat;
% [yPred, accuracy, decisionValues] = libsvmpredict(yTesting, xTesting, model); 
% save yPred.mat yPred;
% save decisionValues.mat decisionValues;
% save xTraining.mat xTraining;
% save yTesting.mat yTesting;

% draw ROC
[totalScores, index]  = sort(decisionValues);
disp(decisionValues(1:10));
disp(totalScores(1:10))
labels = yTesting;
for i = 1 : length(labels)
    labels(i) = yTesting(index(i));
end

truePositive = zeros(1, length(totalScores) + 1);
trueNegative = zeros(1, length(totalScores) + 1);
falsePositive = zeros(1, length(totalScores) + 1);
falseNegative = zeros(1, length(totalScores) + 1);

for i = 1 : length(totalScores)
    if labels(i) == 1
        truePositive(1) = truePositive(1) + 1;
    else
        falsePositive(1) = falsePositive(1) + 1;
    end
end

for i = 1 : length(totalScores)
   if labels(i) == 1
       truePositive(i + 1) = truePositive(i) - 1;
       falsePositive(i + 1) = falsePositive(i);
   else
       falsePositive(i + 1) = falsePositive(i) - 1;
       truePositive(i + 1) = truePositive(i);
   end
end
truePositive = truePositive / truePositive(1);
falsePositive = falsePositive / falsePositive(1);
plot(falsePositive, truePositive);

