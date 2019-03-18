clear;
close all;
clc;
if(isempty(gcp('nocreate')))
    parpool;
end

run matconvnet/matlab/vl_setupnn ;

addpath('pretraining');
addpath('tracking');
addpath('utils');
addpath('rnn');
base_path='/media/cjh/datasets/tracking/OTB100/';
videos = choose_video(base_path);

clc;
close all;
%    if exist([videos{i} '_res.txt'], 'file')
%        continue;
%    end
conf = genConfig('otb',videos);

switch(conf.dataset)
 case 'otb'
    net = fullfile('/media/cjh/cvpaper/git/models/sanet_models','CNN.mat');    % CNNs weights
    rnet = fullfile('/media/cjh/cvpaper/git/models/sanet_models','RNT.mat');   % RNNs weights
end

result = sanet_run(conf.imgList, conf.gt(1,:), net, rnet);

%dlmwrite([videos{i} '_res.txt'], result);