function nn = nn_create(varargin)
SIZE = varargin{1};
nn.size                 =               SIZE;
nn.depth                =               numel(SIZE);
nn.active_function      =               'sigmoid';
nn.output_function      =               'sigmoid';
nn.learning_rate        =               1.5;
nn.weight_decay         =               0;
nn.cost                 =               [];
nn.encoder              =               0;
nn.sparsity             =               0.03;
nn.beta                 =               3;
nn.batch_normalization  =               0;
nn.grad_squared         =               0;
nn.r                    =               0
nn.optimization_method  =               'normal';
nn.objective_function   =               'MSE';



for i = 2:length(varargin)
    if strcmp('active function',varargin{i})
        nn.active_function = varargin{i+1};
    elseif strcmp('output function',varargin{i})
        nn.output_function = varargin{i+1};
    elseif strcmp('learning rate',varargin{i})
        nn.learning_rate = varargin{i+1};
    elseif strcmp('weight decay',varargin{i})
        nn.weight_decay = varargin{i+1};
    elseif strcmp('sparsity',varargin{i})
        nn.sparsity = varargin{i+1};
    elseif strcmp('beta',varargin{i})
        nn.weight_decay = varargin{i+1};
    elseif strcmp('batch normalization',varargin{i})
        nn.batch_normalization = varargin{i+1};
    elseif strcmp('optimization method',varargin{i})
        nn.optimization_method = varargin{i+1};
    elseif strcmp('objective function', varargin{i})
        nn.objective_function = varargin{i+1};
    elseif strcmp('weight decay', varargin{i})
        nn.weight_decay = varargin{i+1};
    end;
end;

if strcmp(nn.objective_function, 'Cross Entropy')
    nn.output_function = 'softmax';
end;

for k = 1 : nn.depth-1
    width = nn.size(k);
    height = nn.size(k+1);
    %nn.W{k} = (rand(height, width) - 0.5) * 2 * sqrt(6 / (height + width + 1)) - sqrt(6 / (height + width + 1));
    
    nn.W{k} = 2*rand(height, width)/sqrt(width)-1/sqrt(width);
    %nn.W{k} = 2*rand(height, width)-1;
    %Xavier initialization 
    if strcmp(nn.active_function, 'relu')
        nn.b{k} = rand(height,1)+0.01;
    else
        nn.b{k} = 2*rand(height, 1)/sqrt(width)-1/sqrt(width);
    end;
    
        
    %parameters for moments
    if strcmp(nn.optimization_method,'Momentum')
        nn.vW{k} = zeros(height,width);
        nn.vb{k} = zeros(height,1);
    end; 
    if strcmp(nn.optimization_method,'AdaGrad')  ||strcmp(nn.optimization_method,'RMSProp') || strcmp(nn.optimization_method,'Adam')
        nn.rW{k} = zeros(height,width);
        nn.rb{k} = zeros(height,1);
    end;
    if strcmp(nn.optimization_method,'Adam')
        nn.sW{k} = zeros(height,width);
        nn.sb{k} = zeros(height,1);
    end; 
       %parameters for batch normalization.
    if nn.batch_normalization
        nn.E{k} = zeros(height,1);
        nn.S{k} = zeros(height,1);
        nn.Gamma{k} = 1;
        nn.Beta{k} = 0;
        if  strcmp(nn.optimization_method,'Momentum')
            nn.vGamma{k} = 1;
            nn.vBeta{k} = 0;
        end;
        if strcmp(nn.optimization_method,'AdaGrad')  ||strcmp(nn.optimization_method,'RMSProp') || strcmp(nn.optimization_method,'Adam')
            nn.rW{k} = zeros(height,width);
            nn.rb{k} = zeros(height,1);
            nn.rGamma{k} = 0;
            nn.rBeta{k} = 0;
        end;
        if  strcmp(nn.optimization_method,'Adam')
            nn.sGamma{k} = 1;
            nn.sBeta{k} = 0;
        end;
    
        nn.vecNum = 0;
    end;
    nn.W_grad{k} = zeros(height,width);
end
