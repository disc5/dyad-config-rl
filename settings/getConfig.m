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
    
    cfg.SIMILARITY_SSIM = 1;
    cfg.SIMILARITY_FSIM = 2;
    cfg.SIMILARTIY_MANHATTAN = 3;
    
    cfg.SAMPLING_PAPI = 1;
    cfg.SAMPLING_PBPI = 2;
    

    %% Configuration
    % Pipeline
    cfg.max_opchain_length = 4;

    % Learning
    cfg.num_rounds = 8; 
    cfg.num_samples_per_round = 100;
    cfg.boltzmann_exploration = true;
    cfg.boltzmann_schedule = [0.01,0.05,0.1,0.5,1,10,100];
    
    % Sampling-Schema
    cfg.sampling_schema = cfg.SAMPLING_PBPI;
    
    % Policy-Model
    cfg.model_type = cfg.MODEL_PLNET;
    cfg.model_state_representation = cfg.STATE_OPERATOR_POSITION;
    cfg.model_action_representation = cfg.ACTION_JOINT_ONEOFK;
    
    % Similarity Measure
    cfg.similarity_measure = cfg.SIMILARITY_SSIM;
end

