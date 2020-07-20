function [wrongs,success_ratio,nn] = nn_test(nn,test_x,test_y)
    nn = nn_predict(nn,test_x);
    y_output = nn.a{nn.depth};
    y_output = y_output';
    [~,label] = max(y_output,[],2); %#ok<*UDIM>
    [~,expection] = max(test_y,[],2);
    wrongs = find(label ~= expection);
    success_ratio = 1-numel(wrongs)/size(test_y,1);
end
    