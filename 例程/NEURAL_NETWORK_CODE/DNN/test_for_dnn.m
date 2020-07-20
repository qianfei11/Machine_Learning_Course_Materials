% clear;clc;close all;
% %addpath(genpath('F:\mayanzhao\M_DeepLearing'));
% load mnist_uint8;
% train_x = double(train_x) / 255;
% test_x  = double(test_x)  / 255;
% train_y = double(train_y);
% test_y  = double(test_y);
% 
% 
% %create a net
% dnn = nn_create([784,400,169,49,10]);
% 
% % train
% dnn = dnn_train(dnn,train_x,train_y);


%adjust
% option.batch_size = 100;
% option.iteration = 10;
% 
% %divide validation set
% [M,N] = size(train_x);
% p = randperm(M); 
% ratioTraining = 0.9;
% ratioValidation = 0.1;
% xTraining = zeros(N,floor(ratioTraining*M));
% [U,V] = size(train_y);
% yTraining = zeros(V,floor(ratioTraining*M));;
% for i=1:floor(ratioTraining*M)
%     xTraining(:,i)  = train_x(p(i),:)';
%     yTraining(:,i) = train_y(p(i),:)';
% end;
% xTraining = xTraining';
% yTraining = yTraining';
% 
% xValidation = zeros(N,M-floor(ratioTraining*M));
% [U,V] = size(train_y);
% yValidation = zeros(V,M-floor(ratioTraining*M));;
% for i=floor(ratioTraining*M)+1:M
%     xValidation(:,i-floor(ratioTraining*M))  = train_x(p(i),:)';
%     yValidation(:,i-floor(ratioTraining*M)) = train_y(p(i),:)';
% end;
% xValidation = xValidation';
% yValidation = yValidation';
% 
% maxAccuracy = 0;
% totalAccuracy = [];
iteration = 0;
while(1)
    iteration = iteration + 1;
    dnn = nn_train(dnn,option,xTraining,yTraining);
    totalCost(iteration) = sum(dnn.cost)/length(dnn.cost);
   % plot(totalCost);
    [wrongs,accuracy] = nn_test(dnn,xValidation,yValidation);
    totalAccuracy = [totalAccuracy,accuracy];
    if accuracy>maxAccuracy
        maxAccuracy = accuracy;
        storedNN = dnn;
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



%adjust
dnn = dnn_adjust(dnn,train_x,train_y);
%test
[wrongs,success_ratio,dnn] = nn_test(dnn,test_x,test_y);
disp(['success rate is ',num2str(success_ratio)]);
figure;
visualize(dnn.W{1}');
figure;
visualize(dnn.W{2}');