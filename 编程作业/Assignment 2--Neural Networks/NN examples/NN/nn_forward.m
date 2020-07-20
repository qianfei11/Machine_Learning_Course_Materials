function nn = nn_forward(nn,batch_x,batch_y)    
    s = size(nn.cost) + 1;
    batch_x = batch_x';
    batch_y = batch_y';
    m = size(batch_x,2);
    nn.a{1} = batch_x;

    cost2 = 0;
    for k = 2 : nn.depth
        y = nn.W{k-1} * nn.a{k-1} + repmat(nn.b{k-1},1,m);
        if nn.batch_normalization
            nn.E{k-1} = nn.E{k-1}*nn.vecNum + sum(y,2);
            nn.S{k-1} = nn.S{k-1}.^2*(nn.vecNum-1) + (m-1)*std(y,0,2).^2;
            nn.vecNum = nn.vecNum + m;
            nn.E{k-1} = nn.E{k-1}/nn.vecNum;
            nn.S{k-1} = sqrt(nn.S{k-1}/(nn.vecNum-1));
            y = (y - repmat(nn.E{k-1},1,m))./repmat(nn.S{k-1}+0.0001*ones(size(nn.S{k-1})),1,m);
            y = nn.Gamma{k-1}*y+nn.Beta{k-1};
        end;
        if k == nn.depth
            switch nn.output_function
                case 'sigmoid'
                    nn.a{k} = sigmoid(y);
                case 'tanh'
                    nn.a{k} = tanh(y);
                case 'relu'
                    nn.a{k} = max(y,0);
                case 'softmax'
                    nn.a{k} = softmax(y);
            end
        else
            switch nn.active_function
                case 'sigmoid'
                    nn.a{k} = sigmoid(y);
                case 'tanh'
                    nn.a{k} = tanh(y);
                case 'relu'
                    nn.a{k} = max(y,0);
            end
        end
        cost2 = cost2 +  sum(sum(nn.W{k-1}.^2));
    end
    if nn.encoder == 1
        roj = sum(nn.a{2},2)/m;
        nn.cost(s) = 0.5 * sum(sum((nn.a{k} - batch_y).^2))/m + 0.5 * nn.weight_decay * cost2 + 3 * sum(nn.sparsity * log(nn.sparsity ./ roj) + ...
            (1-nn.sparsity) * log((1-nn.sparsity) ./ (1-roj)));
    else
        if strcmp(nn.objective_function,'MSE')
            nn.cost(s) = 0.5 / m * sum(sum((nn.a{k} - batch_y).^2)) + 0.5 * nn.weight_decay * cost2;
        elseif strcmp(nn.objective_function,'Cross Entropy')
            nn.cost(s) = -0.5*sum(sum(batch_y.*log(nn.a{k})))/m + 0.5 * nn.weight_decay * cost2;
        %nn.cost(s)
    end
    
end