function [ ] = sanet_pretrain( varargin )

% The list of tracking sequences for training sanet.
opts.seqsList  = {struct('dataset','vot2013','list','path to the root\pretraining\seqList\vot13-otb.txt'),...
    struct('dataset','vot2014','list','path to the root\pretraining\seqList\vot14-otb.txt'),...
    struct('dataset','vot2015','list','path to the root\pretraining\seqList\vot15-otb.txt')};

% The path to the initial network. 
opts.netFile    = fullfile('path to the root\models','sanet_init.mat') ;

% The path to the output
opts.outFile     = fullfile('path to the root\models','CNN.mat') ;
opts.outFileR    = fullfile('path to the root\models','RNT.mat');

% The directory to store the RoIs for training
opts.roiDir     = fullfile('path to the root\models','data_vot-otb') ;

opts.train.batch_frames     = 8 ; % the number of frames to construct a minibatch.
opts.train.batchSize        = 128 ;
opts.train.batch_pos        = 32;
opts.train.batch_neg        = 96;

opts.train.numCycles        = 50 ; % #cycles (#iterations/#domains)
opts.train.useGpu           = true ;
opts.train.conserveMemory	= true ;
opts.train.learningRate     = 0.0001 ; % x10 for fc4-6

opts.sampling.crop_mode         = 'warp';
opts.sampling.numFetchThreads   = 8 ;
opts.sampling.posRange          = [0.7 1];
opts.sampling.negRange          = [0 0.5];
opts.sampling.input_size        = 107;
opts.sampling.crop_padding      = 16;

opts.sampling.posPerFrame       = 50;
opts.sampling.negPerFrame       = 200;
opts.sampling.scale_factor      = 1.05;
opts.sampling.flip              = false;

opts = vl_argparse(opts, varargin) ;
opts.roiPath  = fullfile(opts.roiDir, 'roidb.mat');
genDir(opts.roiDir) ;

%% initial parameters for RNNs
rnn_param.initLr = 0;
rnn_param.lrdecay = 0.9;
rnn_param.momentum = 0.9;
rnn_param.threshold = 2e3;
rnn_param.nx = 96;   % dimension of input neurons   
rnn_param.nh = 96;   % dimension of hidden neurons
rnn_param.ny = 96;   % output
rnn = rnn_initialize(rnn_param);

rnn_param1.initLr = 0;
rnn_param1.lrdecay = 0.9;
rnn_param1.momentum = 0.9;
rnn_param1.threshold = 2e3;
rnn_param1.nx = 256;   % dimension of input neurons   
rnn_param1.nh = 256;   % dimension of hidden neurons
rnn_param1.ny = 256;   % output
rnn1 = rnn_initialize(rnn_param1);

rnn_param2.initLr = 0;
rnn_param2.lrdecay = 0.9;
rnn_param2.momentum = 0.9;
rnn_param2.threshold = 2e3;
rnn_param2.nx = 512;   % dimension of input neurons   
rnn_param2.nh = 512;   % dimension of hidden neurons
rnn_param2.ny = 512;   % output
rnn2 = rnn_initialize(rnn_param2);

%% Sampling training data
if exist(opts.roiPath,'file')
    load(opts.roiPath) ;
else
    roidb = sanet_setup_data(opts.seqsList, opts.sampling);
    save(opts.roiPath, 'roidb') ;
end

K = 1;
net = sanet_init_train(opts, K);

% RNNs
layers_rnn = net.layers{8};
layers_rnn.type = 'rnn';
layers_rnn.name = 'RNNs-Pooling1';

layers_rnn1 = net.layers{8};
layers_rnn1.type = 'rnn1';
layers_rnn1.name = 'RNNs-Pooling2';

layers_rnn2 = net.layers{8};
layers_rnn2.type = 'rnn2';
layers_rnn2.name = 'RNNs-Pooling3';

% pooling3 layer
pooling3 = net.layers{8};
pooling3.name = 'pool3';
pooling3.stride = [1 1];
pooling3.pad = 0;
pooling3.type = 'pool';
pooling3.method = 'max';
pooling3.pool = [1 1];
net.layers = [net.layers(1:4) layers_rnn net.layers(5:8) layers_rnn1 net.layers(9:10) pooling3 layers_rnn2 net.layers(11:end)];

% add rnn to net
net.rnn = rnn;
net.rnn_param = rnn_param;

net.rnn1 = rnn1;
net.rnn_param1 = rnn_param1;

net.rnn2 = rnn2;
net.rnn_param2 = rnn_param2;

fn = @(roidb,img_idx,batch_pos,batch_neg)...
    getBatch(roidb, img_idx, batch_pos, batch_neg, opts.sampling) ;

net = sanet_train(net, roidb, fn, opts.train) ;

%% Save
net = sanet_finish_train(net);
layers = net.layers;

RNT.rnn = net.rnn;
RNT.rnn_param = net.rnn_param;

RNT.rnn1 = net.rnn1;
RNT.rnn_param1 = net.rnn_param1;

RNT.rnn2 = net.rnn2;
RNT.rnn_param2 = net.rnn_param2;

genDir(fileparts(opts.outFile)) ;
save(opts.outFile, 'layers') ;
save(opts.outFileR, 'RNT');

% -------------------------------------------------------------------------
function [im,labels] = getBatch(roidb, img_idx, batch_pos, batch_neg, opts)
% -------------------------------------------------------------------------
image_paths = {roidb(img_idx).img_path};

pos_boxes = cell2mat({roidb(img_idx).pos_boxes}');
idx = randsample(size(pos_boxes,1),batch_pos);
pos_boxes = pos_boxes(idx,:);
pos_idx = floor((idx-1)/opts.posPerFrame)+1;

neg_boxes = cell2mat({roidb(img_idx).neg_boxes}');
idx = randsample(size(neg_boxes,1),batch_neg);
neg_boxes = neg_boxes(idx,:);
neg_idx = floor((idx-1)/opts.negPerFrame)+1;

boxes = [pos_idx, pos_boxes; neg_idx, neg_boxes];

im = get_batch(image_paths, boxes, opts, ...
    'prefetch', nargout == 0) ;

if(nargout > 0 && opts.flip)
    flip_idx = find(randi([0 1],size(boxes,1),1));
    for i=flip_idx
        im(:,:,:,i) = flip(im(:,:,:,i),2);
    end
end

labels = single([2*ones(numel(pos_idx),1);ones(numel(neg_idx),1)]);



% -------------------------------------------------------------------------
function [ roidb ] = sanet_setup_data(seqList, opts)
% -------------------------------------------------------------------------

roidb = {};
for D = 1:length(seqList)
    
    dataset = seqList{D}.dataset;
    seqs_train = importdata(seqList{D}.list);
    
    roidb_ = cell(1,length(seqs_train));
    
    for i = 1:length(seqs_train)
        seq = seqs_train{i};
        fprintf('sampling %s:%s ...\n', dataset, seq);
        
        config = genConfig(dataset, seq);
        roidb_{i} = seq2roidb(config, opts);

    end
    roidb = [roidb,roidb_];
end



% -------------------------------------------------------------------------
function [ net ] = sanet_init_train( opts, K )
% -------------------------------------------------------------------------
net = load(opts.netFile);
net.layers = net.layers(1:end-2);

% domain-specific layers
net.layers{end+1} = struct('type', 'conv', ...
    'name', 'fc6', ...
    'filters', 0.01 * randn(1,1,512,2*K,'single'), ...
    'biases', zeros(1, 2*K, 'single'), ...
    'stride', 1, ...
    'pad', 0, ...
    'filtersLearningRate', 10, ...
    'biasesLearningRate', 20, ...
    'filtersWeightDecay', 1, ...
    'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss_k', 'name', 'loss') ;



% -------------------------------------------------------------------------
function [ net ] = sanet_finish_train( net )
% -------------------------------------------------------------------------
net.layers = net.layers(1:end-2);
for i=1:numel(net.layers)
    switch (net.layers{i}.type)
        case 'conv'
            net.layers{i}.filtersLearningRate = 1;
            net.layers{i}.biasesLearningRate = 2;
    end
end

% new domain-specific layer
net.layers{end+1} = struct('type', 'conv', ...
    'name', 'fc6', ...
    'filters', 0.01 * randn(1,1,512,2,'single'), ...
    'biases', zeros(1, 2, 'single'), ...
    'stride', 1, ...
    'pad', 0, ...
    'filtersLearningRate', 10, ...
    'biasesLearningRate', 20, ...
    'filtersWeightDecay', 1, ...
    'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;



% -------------------------------------------------------------------------
function genDir(path)
% -------------------------------------------------------------------------
if ~exist(path,'dir')
    mkdir(path);
end

