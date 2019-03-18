clear;
clc;
close all;

sanet_prepare_model;

%% Training sanet using the sequences from {VOT13,14,15}
% for experiments on OTB
sanet_pretrain('seqsList',...
    {struct('dataset','vot2013','list','path to the root\pretraining\seqList\vot13-otb.txt'),...
    struct('dataset','vot2014','list','path to the root\pretraining\seqList\vot14-otb.txt'),...
    struct('dataset','vot2015','list','path to the root\pretraining\seqList\vot15-otb.txt')},...
    'outFile', fullfile('path to the root\models','CNN.mat'),...
    'roiDir', fullfile('path to the root\models','data_vot-otb'));