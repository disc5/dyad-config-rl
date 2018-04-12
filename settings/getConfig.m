function [ cfg ] = getConfig()
%GETCONFIG Configuration for Dyad-RL Image Op experiments
%   The config will be used in some parts of api_dr.

    %% Constants
    cfg.STATE_IMAGE = 1;
    cfg.STATE_OPERATOR_POSITION = 2;
    
    cfg.ACTION_OPONEOFK_AND_PARAMRAW = 1;
    cfg.ACTION_OPONEOFK_AND_PARAMMAPPED = 2;
    cfg.ACTION_OPONEOFK_AND_PARAMONEOFK = 3;
    cfg.ACTION_JOINT_ONEOFK = 4;
    
    cfg.MODEL_PLNET = 1;
    cfg.MODEL_PLNET_WITH_CNN_WRAPPER = 2;
    
    cfg.SIMILARITY_SSIM = 1;
    cfg.SIMILARITY_FSIM = 2;
    cfg.SIMILARTIY_MANHATTAN = 3;
    
    cfg.SAMPLING_PAPI = 1;
    cfg.SAMPLING_PBPI = 2;
    

    %% Configuration
    % Pipeline
    cfg.max_opchain_length = 7;

    % Learning
    cfg.num_rounds = 9; 
    cfg.num_samples_per_round = 10;%50;
    cfg.boltzmann_exploration = true;
    cfg.boltzmann_schedule = [0.5, 1, 2, 4, 6, 8, 10];
    cfg.gamma_discount_value = 0.9;
    cfg.gamma_discount_schedule = (0.001*ones(1,7) .* 2.^linspace(5,12,7));
    
    % Sampling-Schema
    cfg.sampling_schema = cfg.SAMPLING_PBPI;
    
    % Policy-Model
    cfg.model_type = cfg.MODEL_PLNET;
    cfg.model_state_representation = cfg.STATE_IMAGE;
    cfg.model_action_representation = cfg.ACTION_JOINT_ONEOFK;
    cfg.model_cnn_wrapper_model = 'results/models/fashioncnn.ckpt';
    
    % Similarity Measure
    cfg.similarity_measure = cfg.SIMILARITY_SSIM;
end

