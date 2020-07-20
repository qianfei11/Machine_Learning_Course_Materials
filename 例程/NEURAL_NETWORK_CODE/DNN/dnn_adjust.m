function dnn = dnn_adjust(dnn,train_x,train_y)
    disp('The cnn is adjusting');
    iteration = 2000;
    batch_size = 100;
    m = size(train_x,1);
    num_batches = m / batch_size;
    dnn.learning_rate = 1.0;
    for k = 1 : iteration
        tic;
        kk = randperm(m);
        for l = 1 : num_batches
            batch_x = train_x(kk((l - 1) * batch_size + 1 : l * batch_size), :);
            batch_y = train_y(kk((l - 1) * batch_size + 1 : l * batch_size), :);
            dnn = nn_forward(dnn,batch_x,batch_y);
            dnn = nn_backpropagation(dnn,batch_y);
            dnn = nn_applygradient(dnn);
        end
        t = toc;
        disp(['Iteration ' num2str(k) '/' num2str(iteration) ' : ' num2str(t) ' seconds']);
    end
    figure;
    plot(dnn.cost);grid on;
end