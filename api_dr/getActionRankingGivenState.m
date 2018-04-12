function [ordering, skills] = getActionRankingGivenState(policy_model, state, params)
%GETACTIONRANKINGGIVENSTATE Determines a ranking of actions.
%   Creates a ranking of parameters(actions) for a given state
%
%   Input:
%       policy_model - the representation of a policy model
%       state - for example an image or the operator position (or both)
%       params - miscellaneous parameters
%
%   Output:
%       ordering - the indices of actions in preferential order, i.e.
%       (ordering(1) corresponds to the ID of an action that is preferred over all other actions) 
%
% (C) 2018 Dirk Schaefer

    [ cfg ] = getConfig();
    
    JointConfigurationSpace = getJointConfigurationSpace();
    skills = zeros(size(JointConfigurationSpace,1),1);
    c = 1000;
    skills2 = zeros(size(JointConfigurationSpace,1),1);
    if cfg.model_type == cfg.MODEL_PLNET
        for i1 = 1 : size(JointConfigurationSpace,1)
            observation = getModelFeatures({state,JointConfigurationSpace(i1,:)});
            [utilities] = policy_model.getUtilities(observation);
            skills(i1) = exp(utilities);
            skills2(i1) = exp(c*utilities);
        end
    elseif cfg.model_type == cfg.MODEL_PLNET_WITH_CNN_WRAPPER
        for i1 = 1 : size(JointConfigurationSpace,1)
            imageFeatures = activations(params.cnn, state, params.cnn_layer, 'OutputAs', 'rows'); % Get Deep CNN FC7 Features
            observation = getModelFeatures({imageFeatures, JointConfigurationSpace(i1,:)});
            [utilities] = policy_model.getUtilities(observation);
            skills(i1) = exp(utilities);
            skills2(i1) = exp(c*utilities);
        end
    else
        error('Other models not implemented yet.');
        ordering = -1;
    end
    [~, ordering] = sort(skills,'descend');
end

