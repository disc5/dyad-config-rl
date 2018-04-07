%% Proof of concept
% Is it possible to represent the preference data with the models?
% Test without actual image features, but operator position ids instead
%
%%
clear all
addpath(genpath('../'))
%%

% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
images = loadMNISTImages('../fashionMNIST/train-images-idx3-ubyte');
labels = loadMNISTLabels('../fashionMNIST/train-labels-idx1-ubyte');

%% Load valid operator ranges
[Op_log_values, Op_gamma_values] = getOperatorParameterSpace();

JointConfigurationSpace = getJointConfigurationSpace();

%% -------------------------------------
%% Data set generation on the first 100 images
%% -------------------------------------
nTrImages = 250;
originals=cell(nTrImages,1);
for i1=1:nTrImages
    id = i1;
    img = images(:,id); % values between 0 and 1
    I=reshape(img,28,28);
    originals{i1} = I;
end


%%
[Op1_Values, Op2_Values] = getOperatorParameterSpace();

%% Construction of  chains using two base operators (with fixed order)
% i.e., Op1 (log) and Op2 (gamma)
% Op1->Op1->Op2->Op1 with random parameter values
% thus the inverse sequence is Op1<-Op2<-Op1<-Op1
%
distorted=cell(nTrImages,1);
chain_prefs_tr = cell(nTrImages,1);
params = cell(nTrImages,4);
for i1=1:nTrImages
    chain_preferences = cell(0,1);
    chain_pref_count = 1;
    
    id = i1;
    img = images(:,id);     % values between 0 and 1
    I=reshape(img,28,28);
    
    I_current = I;
  
    

    current_operator_position = 1;
    rparam_val = 2.5;%Op_log_values(randi(length(Op_log_values)));
    params{i1,4} = rparam_val;
    [I_current] = applyInvOperator(1, rparam_val, I_current);
    
    % Set pairwise preferences systematically
    for i2 = 1 : length(Op1_Values)
        if Op1_Values(i2)~=rparam_val
            %ct_rand_action = JointConfigurationSpace(randi(size(JointConfigurationSpace,1)),:);
            ct_rand_op_id = 1;
            ct_rand_op_params = Op1_Values(i2);
            %ct_state =  activations(net,imresize(ct_state(:,:,[1 1 1]),[inputSize(1) inputSize(2)],'nearest'),'fc7','OutputAs','rows');
            chain_preferences{chain_pref_count,1} = {[1,rparam_val, current_operator_position]}; % winner
            chain_preferences{chain_pref_count,2} = {[ct_rand_op_id, ct_rand_op_params, current_operator_position]}; % looser
            chain_pref_count = chain_pref_count + 1;
        end
    end
    for i2 = 1 : length(Op2_Values)
            ct_rand_op_id = 2;
            ct_rand_op_params = Op2_Values(i2);
            %ct_state =  activations(net,imresize(ct_state(:,:,[1 1 1]),[inputSize(1) inputSize(2)],'nearest'),'fc7','OutputAs','rows');
            chain_preferences{chain_pref_count,1} = {[1,rparam_val, current_operator_position]}; % winner
            chain_preferences{chain_pref_count,2} = {[ct_rand_op_id, ct_rand_op_params, current_operator_position]}; % looser
            chain_pref_count = chain_pref_count + 1;
    end
    
    
    current_operator_position = 2;
    rparam_val = 1.4;%Op_gamma_values(randi(length(Op_gamma_values)));
    params{i1,3} = rparam_val;
    [I_current] = applyInvOperator(2, rparam_val, I_current);
    
    % Set pairwise preferences systematically
    for i2 = 1 : length(Op1_Values)
        ct_rand_op_id = 1;
        ct_rand_op_params = Op1_Values(i2);
        chain_preferences{chain_pref_count,1} = {[2,rparam_val, current_operator_position]}; % winner
        chain_preferences{chain_pref_count,2} = {[ct_rand_op_id, ct_rand_op_params, current_operator_position]}; % looser
        chain_pref_count = chain_pref_count + 1;

    end
    for i2 = 1 : length(Op2_Values)
         if Op2_Values(i2)~=rparam_val
            ct_rand_op_id = 2;
            ct_rand_op_params = Op2_Values(i2);
            chain_preferences{chain_pref_count,1} = {[2,rparam_val,current_operator_position]}; % winner
            chain_preferences{chain_pref_count,2} = {[ct_rand_op_id, ct_rand_op_params, current_operator_position]}; % looser
            chain_pref_count = chain_pref_count + 1;
         end
    end
 
   
    current_operator_position = 3;
    rparam_val = 1.5;%Op_log_values(randi(length(Op_log_values)));
    params{i1,2} = rparam_val;
    [I_current] = applyInvOperator(1, rparam_val, I_current);
    
    % Set pairwise preferences systematically
    for i2 = 1 : length(Op1_Values)
        if Op1_Values(i2)~=rparam_val
            ct_rand_op_id = 1;
            ct_rand_op_params = Op1_Values(i2);
            chain_preferences{chain_pref_count,1} = {[1,rparam_val,current_operator_position]}; % winner
            chain_preferences{chain_pref_count,2} = {[ct_rand_op_id, ct_rand_op_params,current_operator_position]}; % looser
            chain_pref_count = chain_pref_count + 1;
        end
    end
    for i2 = 1 : length(Op2_Values)
            ct_rand_op_id = 2;
            ct_rand_op_params = Op2_Values(i2);
            chain_preferences{chain_pref_count,1} = {[1,rparam_val,current_operator_position]}; % winner
            chain_preferences{chain_pref_count,2} = {[ct_rand_op_id, ct_rand_op_params,current_operator_position]}; % looser
            chain_pref_count = chain_pref_count + 1;
    end
    
    
    
    current_operator_position = 4;
    rparam_val = 2.0;
    params{i1,1} = rparam_val;
    [I_current] = applyInvOperator(1, rparam_val, I_current);
    
    % Set pairwise preferences systematically
    for i2 = 1 : length(Op1_Values)
        if Op1_Values(i2)~=rparam_val
            %ct_rand_action = JointConfigurationSpace(randi(size(JointConfigurationSpace,1)),:);
            ct_rand_op_id = 1;
            ct_rand_op_params = Op1_Values(i2);
            chain_preferences{chain_pref_count,1} = {[1,rparam_val,current_operator_position]}; % winner
            chain_preferences{chain_pref_count,2} = {[ct_rand_op_id, ct_rand_op_params,current_operator_position]}; % looser
            chain_pref_count = chain_pref_count + 1;
        end
    end
    for i2 = 1 : length(Op2_Values)
            ct_rand_op_id = 2;
            ct_rand_op_params = Op2_Values(i2);
            chain_preferences{chain_pref_count,1} = {[1,rparam_val,current_operator_position]}; % winner
            chain_preferences{chain_pref_count,2} = {[ct_rand_op_id, ct_rand_op_params,current_operator_position]}; % looser
            chain_pref_count = chain_pref_count + 1;
    end
    
    chain_prefs_tr{i1} = chain_preferences;
    
          
    distorted{i1} = I_current;
end

% ------------------------------------------------------------------------
%% LEARNING Phase
[netData] = convertRoundOpChainPreferencesToTrainingData_OpPos_PLNet(chain_prefs_tr);

%% 
clear mynet;
cfg = getConfig();
mynet = plnet([8+cfg.max_opchain_length,10,1],0.1); % 4096
mynet.SGD(netData, 10, 0.1); %200
%     
policy_model = mynet;
    
%%  (3) evaluate policy
[total_error_tr] = evaluatePolicy_PLNet(policy_model, distorted, originals);
total_error_tr
