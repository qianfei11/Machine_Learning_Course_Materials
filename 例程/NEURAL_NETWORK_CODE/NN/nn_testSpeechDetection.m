% clear all;
% A = importdata('training.DATA');
% 
% flag = 1;
% while flag<=size(A,1)
%     whetherNan = 0;
%     for j = 1:size(A,2)
%         if isnan(A(flag,j))
%             whetherNan = 1;
%             break;
%         end;
%     end;
%     if whetherNan
%         A(flag,:) = [];
%     else
%         flag = flag+1;
%     end;
% end;
% 
% newA = importdata('testing.DATA');
% 
% flag = 1;
% while flag<=size(newA,1)
%     whetherNan = 0;
%     for j = 1:size(newA,2)
%         if isnan(newA(flag,j))
%             whetherNan = 1;
%             break;
%         end;
%     end;
%     if whetherNan
%         newA(flag,:) = [];
%     else
%         flag = flag+1;
%     end;
% end;
% 
% A = [A',newA']';
% [M,N] = size(A);
% 
% xapp = zeros(N-1,M);
% yapp = zeros(2,M);
% for i = 1:M
%     xapp(:,i) = A(i,[1:N-1])';
%     if A(i,N) >0
%         yapp(1,i) = 1;
%     else
%         yapp(2,i) = 1;
%     end;
% end;
% 
% ratioTraining = 0.4;
% ratioValidation = 0.1;
% ratioTesting = 0.5;
% xTraining = [];
% yTraining = [];
% p = randperm(M); 
% for i=1:floor(ratioTraining*M)
%     xTraining  = [xTraining,xapp(:,p(i))];
%     yTraining = [yTraining,yapp(:,p(i))];
% end;
% xTraining = xTraining';
% yTraining = yTraining';
% 
% 
% [U,V] = size(xTraining);
% avgX = mean(xTraining);
% sigma = std(xTraining);
% xTraining = (xTraining - repmat(avgX,U,1))./repmat(sigma,U,1);
% 
% xValidation = [];
% yValidation = [];
% for i=floor(ratioTraining*M)+1:floor((ratioTraining+ratioValidation)*M)
%     xValidation  = [xValidation,xapp(:,p(i))];
%     yValidation = [yValidation,yapp(:,p(i))];
% end;
% xValidation= xValidation';
% yValidation = yValidation';
% 
% [U,V] = size(xValidation);
% xValidation = (xValidation - repmat(avgX,U,1))./repmat(sigma,U,1);
% 
% xTesting = [];
% yTesting = [];
% for i=floor((ratioTraining+ratioValidation)*M)+1:M
%     xTesting  = [xTesting,xapp(:,p(i))];
%     yTesting = [yTesting,yapp(:,p(i))];
% end;
% xTesting = xTesting';
% yTesting = yTesting';
% [U,V] = size(xTesting);
% xTesting = (xTesting - repmat(avgX,U,1))./repmat(sigma,U,1);



%create a netual net
clear nn;
nn = nn_create([6,10,10,10,10,10,10,10,10,10,10,10,2],'active function','tanh','learning rate',0.1, 'batch normalization',1,'optimization method','normal');

%train
option.batch_size = 100;
option.iteration = 10;

ratioPrevious = 0;
flag = 1;
delta = 0.01;
iteration = 0;
maxAccuracy = 0;
totalAccuracy = [];
while(1)
    iteration = iteration +1; 
    nn = nn_train(nn,option,xTraining,yTraining);
    totalCost(iteration) = sum(nn.cost)/length(nn.cost);
   % plot(totalCost);
    [wrongs,accuracy] = nn_test(nn,xValidation,yValidation);
    totalAccuracy = [totalAccuracy,accuracy];
    if accuracy>maxAccuracy
        maxAccuracy = accuracy;
        storedNN = nn;
    end;
    cost = totalCost(iteration);
    accuracy
    cost
%     if mod(iteration,10) == 0
%         subplot(2,1,1);
%         title('Average Objective Function Value on the Training Set');
%         plot(totalCost);
% 
%         subplot(2,1,2);
%         title('Accuracy on the Validation Set');
%         plot(totalAccuracy);
%     end;

end;
[wrongs,accuracy] = nn_test(storedNN,xTesting,yTesting);

%testing

%load storedNN.mat;
A = importdata('testing.DATA');

flag = 1;
while flag<=size(A,1)
    whetherNan = 0;
    for j = 1:size(A,2)
        if isnan(A(flag,j))
            whetherNan = 1;
            break;
        end;
    end;
    if whetherNan
        A(flag,:) = [];
    else
        flag = flag+1;
    end;
end;

[M,N] = size(A);

xTesting = zeros(N-1,M);
yTesting = zeros(2,M);
for i = 1:M
    xTesting(:,i) = A(i,[1:N-1])';
    if A(i,N) >0
        yTesting(1,i) = 1;
    else
        yTesting(2,i) = 1;
    end;
end;
xTesting = xTesting';
yTesting = yTesting';
xTesting = (xTesting - repmat(avgX,M,1))./repmat(sigma,M,1);
[wrongs,accuracy] = nn_test(storedNN,xTesting,yTesting);