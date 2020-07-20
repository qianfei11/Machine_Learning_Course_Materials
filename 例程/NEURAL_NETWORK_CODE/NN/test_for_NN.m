clear;clc;close all;
addpath(genpath('D:\matlab\NN example autoencoder\data'));
load mnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);


% normalize


%create a netual net
nn = nn_create([784,40,10]);
%train
option.batch_size = 100;
option.iteration = 5;
nn = nn_train(nn,option,train_x,train_y);
%test
[wrongs,ratio] = nn_test(nn,test_x,test_y);
disp([num2str(size(test_x,1)) ' photos have been tested, the success ratio is ' num2str(ratio)]);
