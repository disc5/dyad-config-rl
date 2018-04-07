%% Visualize model trained with Workflow_X.m

%% Preliminaries
clear all;
%basePath = 'D:\Dropbox\code\mylibs';
basePath = '/Users/dschaefer/Dropbox/code/mylibs';
addpath(genpath(basePath));
addpath('../image_operators/')
addpath('../ext/')
addpath('../helper')
addpath('../evaluation')
addpath('../')

%% Load Model
load('../models/policy_model2.mat');

%% Load data set
load('../data/fashion-mnist-distort100.mat')
nDataSamples = size(originals,1);

load('../data/fashion-mnist-test-distort100.mat')
nTeDataSamples = size(te_originals,1);

%%========================================================================
%% Policy Inspection
JointConfigurationSpace = getJointConfigurationSpace();
[total_error] = evaluatePolicy_PLNet(policy_model, te_distorted, te_originals, JointConfigurationSpace)



%% Inspect multiple images
%selected_ids = [4,7,8,9,16,25,40,42,43,62];
selected_ids = [4,7,8,16,25,42,43,62];

[results] = benchmarkMultipleAndPlot(policy_model, distorted, originals, selected_ids)
x0=150;
y0=550;
width=850;
height=800;
set(gcf,'units','points','position',[x0,y0,width,height])
%% Inspect single image
%id = 4;
%[result] = benchmarkModel(policy_model, te_distorted{id}, te_originals{id})