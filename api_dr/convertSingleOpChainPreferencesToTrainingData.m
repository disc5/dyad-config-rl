function [preferenceData] = convertSingleOpChainPreferencesToTrainingData(chain_comparisons)
%CONVERTTOTRAININGDATA Converts chain preferences into actual training data
%   The functions creates a joint-feature tensor for jf model training.
%
%   Input:
%       chain_comparison - K^2 x 2 cell array of (state,action) choices,
%           where the choice at the first column is preferred over that of
%           the second column
%
%   Output:
%       preferenceData - for example for plnet, a tensor K^2 x 2 x dim (ordered) tensor for model training, e.g.
%           with jfpl_optim_ordered_tensor.m
%
% (C) 2018 Dirk Schaefer

    cfg = getConfig();
    Ksq = size(chain_comparisons,1);
    
    JointConfigurationSpace = getJointConfigurationSpace();
    K = size(JointConfigurationSpace,1);
    
    if cfg.model_type == cfg.MODEL_PLNET || cfg.model_type == cfg.MODEL_PLNET_WITH_CNN_WRAPPER
    
        plNetTensor = zeros(Ksq, 2, length(getModelFeatures())); % 787

        for i1 = 1 : Ksq
            ct_winner_state_action_info = chain_comparisons{i1,1};
            [ct_winner_model_features] = getModelFeatures(ct_winner_state_action_info);
            plNetTensor(i1,1,:) = [ct_winner_model_features];

            ct_looser_state_action_info = chain_comparisons{i1,2};
            [ct_looser_model_features] = getModelFeatures(ct_looser_state_action_info);
            plNetTensor(i1,2,:) = [ct_looser_model_features];
        end

        preferenceData = plNetTensor;
    else
        error('To be implemented.');
        preferenceData = -1;
    end
end

