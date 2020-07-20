% clear all;
% fid  =  fopen('krkopt.DATA');
% c = fread(fid, 3);
% 
% 
% vec = zeros(6,1);
% xapp = [];
% yapp = [];
% while ~feof(fid)
%     string = [];
%     c = fread(fid,1);
%     flag = flag+1;
%     while c~=13
%         string = [string, c];
%         c=fread(fid,1);
%     end;
%     fread(fid,1);  
%     if length(string)>10
%         vec(1) = string(1) - 96;
%         vec(2) = string(3) - 48;
%         vec(3) = string(5) - 96;
%         vec(4) = string(7) - 48;
%         vec(5) = string(9) - 96;
%         vec(6) = string(11) - 48;
%         xapp = [xapp,vec];
%         if string(13) == 100
%             yapp = [yapp,[1,0]'];
%         else
%             yapp = [yapp,[0,1]'];
%         end;
%     end;
% end;
% fclose(fid);
% 
% [N,M] = size(xapp);
% p = randperm(M); 
% ratioTraining = 0.4;
% ratioValidation = 0.1;
% ratioTesting = 0.5;
% xTraining = [];
% yTraining = [];
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
nn = nn_create([6,10,10,10,10,10,10,2],'active function','tanh','learning rate',0.1, 'batch normalization',0,'optimization method','normal', 'objective function', 'Cross Entropy');

%train
option.batch_size = 100;
option.iteration = 10;


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


