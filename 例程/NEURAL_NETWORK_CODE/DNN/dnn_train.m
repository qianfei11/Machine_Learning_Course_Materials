function dnn = dnn_train(dnn,train_x,train_y)
    option.batch_size = 100;
    option.iteration = 10;
    for k = 1 : numel(dnn.size)- 2
        disp(['The ' num2str(k) '/' num2str(numel(dnn.size)-1) ' hidden layer is traing']);
        sae = sae_create([dnn.size(k),dnn.size(k+1)]);
        sae = sae_train(sae,option,train_x);
        dnn.W{k} = sae.W{1};
        dnn.b{k} = sae.b{1};
        sae = nn_predict(sae,train_x);
        train_x = sae.a{2}';
    end
    k = k + 1;
    disp(['The ' num2str(k) '/' num2str(numel(dnn.size)-1) ' hidden layer is traing']);
    nn = nn_create([dnn.size(k),dnn.size(k+1)]);
    nn = nn_train(nn,option,train_x,train_y);
    dnn.W{k} = nn.W{1};
    dnn.b{k} = nn.b{1};
end