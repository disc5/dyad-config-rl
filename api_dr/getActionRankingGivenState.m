function [ordering, skills] = getActionRankingGivenState(policy_model, state)
%GETACTIONRANKINGGIVENSTATE Determines a ranking of actions.
%   Creates a ranking of parameters(actions) for a given state
%
%   Input:
%       policy_model - the representation of a policy model
%       state - for example an image or the operator position (or both)
%
%   Output:
%       ordering - the indices of actions in preferential order, i.e.
%       (ordering(1) corresponds to the ID of an action that is preferred over all other actions) 
%
% (C) 2018 Dirk Schaefer

    [ cfg ] = getConfig();
    
    JointConfigurationSpace = getJointConfigurationSpace();
    skills = zeros(size(JointConfigurationSpace,1),1);
    if cfg.model_type == cfg.MODEL_PLNET
        for i1 = 1 : size(JointConfigurationSpace,1)
            observation = getModelFeatures({state,JointConfigurationSpace(i1,:)});
            [utilities] = policy_model.getUtilities(observation);
            skills(i1) = exp(utilities);
        end

        [~, ordering] = sort(skills,'descend');
    else
        error('Other models not implemented yet.');
        ordering = -1;
    end
    
end

