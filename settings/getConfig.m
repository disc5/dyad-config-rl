function [ cfg ] = getConfig()
%GETCONFIG Configuration for Dyad-RL Image Op experiments
%   The config will be used in some parts of api_dr.

    %% Constants
    cfg.STATE_IMAGE = 1;
    cfg.STATE_OPERATOR_POSITION = 2;
    cfg.MODEL_PLNET = 1;
    cfg.SIMILARITY_SSIM = 1;
    cfg.SIMILARITY_FSIM = 2;
    cfg.SIMILARTIY_MANHATTAN = 3;
    

    %% Configuration
    % Pipeline
    cfg.max_opchain_length = 4;

    % Learning
    cfg.num_rounds = 4; 
    cfg.num_samples_per_round = 50;
    
    % Policy-Model
    cfg.model_type = cfg.MODEL_PLNET;
    cfg.model_state_representation = cfg.STATE_OPERATOR_POSITION;
    
    % Similarity Measure
    cfg.similarity_measure = cfg.SIMILARITY_SSIM;
end

