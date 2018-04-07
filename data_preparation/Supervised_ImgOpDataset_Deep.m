%% Proof of concept
% Is it possible to represent the preference data with the models?
%
%%
clear all
addpath(genpath('../'))

%% Load valid operator ranges
[Op_log_values, Op_gamma_values] = getOperatorParameterSpace();
JointConfigurationSpace = getJointConfigurationSpace();

%% Feature Extraction
net = alexnet
inputSize = net.Layers(1).InputSize

%% -------------------------------------
%% Data set generation on the first 100 images
%% -------------------------------------
nTrImages = 5;
originals=cell(nTrImages,1);
I = imread('../caltech/training/image_0001.jpg');
originals{1} = im2double(I);
originals{2} = im2double(imread('../caltech/training/image_0002.jpg'));
originals{3} = im2double(imread('../caltech/training/image_0003.jpg'));
originals{4} = im2double(imread('../caltech/training/image_0010.jpg'));
originals{5} = im2double(imread('../caltech/training/image_0023.jpg'));


%% Processing
for i1 = 1:nTrImages
    originals{i1} = imresize(originals{i1},[inputSize(1) inputSize(2)],'nearest');
end
%%
%imshow(originals{4})

%% Construction of  chains using two base operators (with fixed order)
% i.e., Op1 (log) and Op2 (gamma)
% Op1->Op1->Op2->Op1 with random parameter values
% thus the inverse sequence is Op1<-Op2<-Op1<-Op1
%
distorted=cell(nTrImages,1);
chain_prefs_tr = cell(nTrImages,1);
params = cell(nTrImages,4);
for i1=1:nTrImages
    chain_preferences = cell(0,2);
    chain_pref_count = 1;
    
    I_current = originals{i1};
    
    ct_state = I_current;
    rparam_val = Op_log_values(randi(length(Op_log_values)));
    params{i1,4} = rparam_val;
    [I_current] = applyInvOperator(1, rparam_val, I_current);
    
    ct_rand_action = JointConfigurationSpace(randi(size(JointConfigurationSpace,1)),:);
    ct_rand_op_id = ct_rand_action(1);
    ct_rand_op_params = ct_rand_action(2:end);
    
    % Check:
%     looser_result = applyOperator(ct_rand_op_id, ct_rand_op_params, I_current);
%     winner_result = applyOperator(1, rparam_val, I_current);
%     figure
%     imshow(looser_result)
%     title('looser')
%     figure
%     imshow(winner_result)
%     title('winner')
%   
    PREV_IMG = I_current;
    I_current =  mat2gray(activations(net,imresize(I_current,[inputSize(1) inputSize(2)],'nearest'),'fc7','OutputAs','rows'));
    chain_preferences{chain_pref_count,1} = {I_current, [1,rparam_val]}; % winner
    chain_preferences{chain_pref_count,2} = {I_current, [ct_rand_op_id, ct_rand_op_params]}; % looser
    chain_pref_count = chain_pref_count + 1;
    
    
    
    I_current=PREV_IMG;
    rparam_val = Op_gamma_values(randi(length(Op_gamma_values)));
    params{i1,3} = rparam_val;
    [I_current] = applyInvOperator(2, rparam_val, I_current);
    
    ct_rand_action = JointConfigurationSpace(randi(size(JointConfigurationSpace,1)),:);
    ct_rand_op_id = ct_rand_action(1);
    ct_rand_op_params = ct_rand_action(2:end);
    PREV_IMG = I_current;
    I_current =  mat2gray(activations(net,imresize(I_current,[inputSize(1) inputSize(2)],'nearest'),'fc7','OutputAs','rows'));
    chain_preferences{chain_pref_count,1} = {I_current, [2,rparam_val]}; % winner
    chain_preferences{chain_pref_count,2} = {I_current, [ct_rand_op_id, ct_rand_op_params]}; % looser
    chain_pref_count = chain_pref_count + 1;
    

    I_current=PREV_IMG;
    rparam_val = Op_log_values(randi(length(Op_log_values)));
    params{i1,2} = rparam_val;
    [I_current] = applyInvOperator(1, rparam_val, I_current);
    
    ct_rand_action = JointConfigurationSpace(randi(size(JointConfigurationSpace,1)),:);
    ct_rand_op_id = ct_rand_action(1);
    ct_rand_op_params = ct_rand_action(2:end);
    PREV_IMG = I_current;
    I_current =  mat2gray(activations(net,imresize(I_current,[inputSize(1) inputSize(2)],'nearest'),'fc7','OutputAs','rows'));
    chain_preferences{chain_pref_count,1} = {I_current, [1,rparam_val]}; % winner
    chain_preferences{chain_pref_count,2} = {I_current, [ct_rand_op_id, ct_rand_op_params]}; % looser
    chain_pref_count = chain_pref_count + 1;
    
    
    I_current=PREV_IMG;
    rparam_val = Op_log_values(randi(length(Op_log_values)));
    params{i1,1} = rparam_val;
    [I_current] = applyInvOperator(1, rparam_val, I_current);
    
    ct_rand_action = JointConfigurationSpace(randi(size(JointConfigurationSpace,1)),:);
    ct_rand_op_id = ct_rand_action(1);
    ct_rand_op_params = ct_rand_action(2:end);
    PREV_IMG = I_current;
    I_current =  mat2gray(activations(net,imresize(I_current,[inputSize(1) inputSize(2)],'nearest'),'fc7','OutputAs','rows'));
    chain_preferences{chain_pref_count,1} = {I_current, [1,rparam_val]}; % winner
    chain_preferences{chain_pref_count,2} = {I_current, [ct_rand_op_id, ct_rand_op_params]}; % looser
    chain_pref_count = chain_pref_count + 1;
    
    chain_prefs_tr{i1} = chain_preferences;
    
          
    distorted{i1} = I_current;
end

% ------------------------------------------------------------------------
%% LEARNING Phase
[netData] = convertRoundOpChainPreferencesToTrainingData_PLNet(chain_prefs_tr);


%%
clear mynet;
mynet = plnet([4096+2+1,10,1],0.1); % 4096
mynet.SGD(netData, 900, 0.1); %200
%     
policy_model = mynet;
    
%%  (3) evaluate policy
%[total_error_tr] = evaluatePolicy_PLNet(policy_model, distorted, originals);
%total_error_tr

%% Alexnet / TEMP
% Test: use alex net features
% to this end, the grayscale image must be embedded into rgb space.
%I(:,:,[1 1 1]) or repmat(YourImage, 1, 1, 3)
net = alexnet
inputSize = net.Layers(1).InputSize
%%
I2 = I(:,:,[1 1 1])
I3 = imresize(I2,[inputSize(1) inputSize(2)],'nearest')
imshow(I3)
%%
layer = 'fc7';
featuresTrain = activations(net,I3,layer,'OutputAs','rows');

%% One-liner
ct_img_features =  activations(net,imresize(I(:,:,[1 1 1]),[inputSize(1) inputSize(2)]),'fc7','OutputAs','rows');

