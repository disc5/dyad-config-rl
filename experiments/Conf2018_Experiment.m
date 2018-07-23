%% Conf2018 Experiment: Image Enhancement - Scenario: "Full-reference-image"
% This script produces the learning curves provided in the paper.
%
%% Preliminaries
clear all;
addpath(genpath('../'))

%% Config
cfgFilename = 'ecml_cfg.json';
cfg = getConfig(cfgFilename);
% %% Configuration
%     % Pipeline
%     cfg.max_opchain_length = 4;
% 
%     % Learning
%     cfg.num_rounds = 10; 
%     cfg.num_samples_per_round = 10;
%     cfg.boltzmann_exploration = true;
%     cfg.boltzmann_schedule = [0.1, 0.5, 1, 2.5, 5, 7.5, 10];
%     
%     % Sampling-Schema
%     cfg.sampling_schema = cfg.SAMPLING_PBPI;
%     
%     % Policy-Model
%     cfg.model_type = cfg.MODEL_PLNET;
%     cfg.model_state_representation = cfg.STATE_OPERATOR_POSITION;
%     cfg.model_action_representation = cfg.ACTION_JOINT_ONEOFK;
%     
%     % Similarity Measure
%     cfg.similarity_measure = cfg.SIMILARITY_SSIM;
%%
params.cfg = cfg;
%% Load training and test data
load('../data/fashion-mnist-distort100-4ch.mat')
nDataSamples = size(originals,1);

load('../data/fashion-mnist-test-distort100-4ch.mat')
nTeDataSamples = size(te_originals,1);


nMultiRuns = 10;
multi_run_results = cell(nMultiRuns,1);

for i100 = 1 : nMultiRuns
    tic
    %% Init Policy Model
    net = plnet([length(getModelFeatures()),10,1],0.1);
    policy_model = net.copy();

    %%
    [total_error_before_tr] = evaluatePolicy(policy_model, distorted, originals,params);
    [total_error_before_te] = evaluatePolicy(policy_model, te_distorted, te_originals,params);

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

            [ct_chain_preferences] = elicitPipelineOperatorPreferences(policy_model, ct_state0, ct_gt_state, i1, params);
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
        while (length(global_training_data) > 65000)
            global_training_data(1)=[]; % remove oldest entry
        end

        fprintf('Size gl_training data: %d\n',size(global_training_data,1));
        %%  (2) perform training: learn next generation policy model
        clear net;
        close all;
        net = plnet([length(getModelFeatures()),10,1],0.1);
        %net = policy_model.copy();
        net.SGD(global_training_data, 20, 0.1);

        policy_model_old = policy_model;
        policy_model = net;

        %%  (3) evaluate policy
        [total_error_tr] = evaluatePolicy(policy_model, distorted, originals, params);
        [total_error_te] = evaluatePolicy(policy_model, te_distorted, te_originals, params);
        fprintf('Error after round %d : Tr=%3.4f \t Te=%3.4f \n', ct_round, total_error_tr, total_error_te);
        pause(2)
        learn_results{i1}.model = policy_model;
        learn_results{i1}.quality = total_error_te;

    end % rounds
    qual = zeros(1, length(learn_results));
    for i1 = 1:length(learn_results)
        fprintf('Quality round %d : %3.4f \n', i1, learn_results{i1}.quality);
        qual(i1) = learn_results{i1}.quality;
    end
    multi_run_results{i100} = [total_error_before_te, qual];
    toc
end % multiple v1

%% Save results
%save('../results/ecml_result.mat','multi_run_results');

%% Evaluate
figure
hold on
all_errors = zeros(10,10);
for i1 = 1 :10
    plot(1:10,multi_run_results{i1},'-o')
    all_errors(i1,:) = multi_run_results{i1};
end

%%
m = mean(all_errors)
s = std(all_errors)
%%
figure
e=errorbar(m,s)
xlabel('Rounds')
ylabel('Error')

e.Color = [0,0,0];
e.LineWidth = 1.5;
e.Marker = 'o';
e.MarkerFaceColor = [1,1,1];
e.MarkerSize = 10;
set(gca, ...
  'Box'         , 'off'     , ...
  'TickDir'     , 'out'     , ...
  'TickLength'  , [.02 .02] , ...
  'XMinorTick'  , 'off'      , ...
  'YMinorTick'  , 'off'      , ...
  'YGrid'       , 'on'      , ...
  'XColor'      , [.3 .3 .3], ...
  'YColor'      , [.3 .3 .3], ...
  'LineWidth'   , 1         );

%%
