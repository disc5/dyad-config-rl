%% Experiment: Image Enhancement - Scenario: "Full-reference-image"
%
% State: op_position within chain (no image)
% Action: Operator+Param as Class
%
% A pairwise preference is generated for a chain position, if an action a1 
% lead to a better endresult after the rollout than an action a2.
% 

%% Preliminaries
clear all;
addpath(genpath('../'))

%% Config
cfg = getConfig();

%% Load training and test data
load('../data/fashion-mnist-distort100-4ch.mat')
nDataSamples = size(originals,1);

load('../data/fashion-mnist-test-distort100-4ch.mat')
nTeDataSamples = size(te_originals,1);

%% Init Policy Model
net = plnet([8+cfg.max_opchain_length,10,1],0.1); % 8: 1-of-k encoding of params
policy_model = net.copy();


[total_error_before_tr] = evaluatePolicy(policy_model, distorted, originals);
[total_error_before_te] = evaluatePolicy(policy_model, te_distorted, te_originals);

fprintf('Error before training : Tr=%3.4f \t Te=%3.4f \n',  total_error_before_tr, total_error_before_te);

% Further book-keeping vars
learn_results = cell(cfg.num_rounds,1);
global_training_data = cell(0,0);

%% ----------------------------------------------------------------------
%% Main-Loop 
for i1 = 1 : cfg.num_rounds
    fprintf('Enter round %d/%d \n', i1, cfg.num_rounds);
    
    % INIT Roundta
    ct_round = i1;
    round_chain_preferences = cell(cfg.num_samples_per_round,1);
    

    % Sample data for current round
    RP=randperm(nDataSamples);
    round_distorted = distorted(RP(1:cfg.num_samples_per_round));
    round_groundtruth = originals(RP(1:cfg.num_samples_per_round));

    %% Simulation phase
    for i2 = 1 : cfg.num_samples_per_round
        ct_state0= round_distorted{i2};
        ct_gt_state = round_groundtruth{i2};

        [ct_chain_preferences] = elicitPipelineOperatorPreferences(policy_model, ct_state0, ct_gt_state);
        round_chain_preferences{i2} = ct_chain_preferences;
        if (mod(i2,80)==1)
            fprintf('\n')
        end
        fprintf('#')
    end      
    fprintf('\n')

    %% At the end of the round:
    %%  (1) generate (dyadic) preferences in the format that can be used to learn the next
    [netData] = convertRoundOpChainPreferencesToTrainingData(round_chain_preferences);

    if isempty(global_training_data) == 1
        global_training_data = netData;
    else
        %global_training_data = [global_training_data; netData];
        global_training_data = [netData];
    end
    
    %% Realize fixed sized data queue
    while (length(global_training_data) > 5000)
        global_training_data(1)=[]; % remove oldest entry
    end
    
    fprintf('Size gl_training data: %d\n',size(global_training_data,1));
    %%  (2) perform training: learn next generation policy model
    clear net;
    close all;
    net = plnet([8+cfg.max_opchain_length,10,1],0.1);
%    net = policy_model.copy();
    net.SGD(global_training_data, 20, 0.1); %200
    
    policy_model_old = policy_model;
    policy_model = net;
    
    %%  (3) evaluate policy
    [total_error_tr] = evaluatePolicy(policy_model, distorted, originals);
    %total_error_tr = 1;
    [total_error_te] = evaluatePolicy(policy_model, te_distorted, te_originals);
    fprintf('Error in round %d : Tr=%3.4f \t Te=%3.4f \n', ct_round, total_error_tr, total_error_te);
    pause(2)
    learn_results{i1}.model = policy_model;
    learn_results{i1}.quality = total_error_te;
    
end % rounds
%%
for i1 = 1:length(learn_results)
    fprintf('Quality round %d : %3.4f \n', i1, learn_results{i1}.quality);
end

%% Save policy model
%save('../results/models/policy_model.mat','policy_model');

%%========================================================================
[total_error_te, te_restored,allIntermediates] = evaluatePolicy(policy_model, te_distorted, te_originals);
showDistRestoredOriginals(te_originals, te_distorted, te_restored)


