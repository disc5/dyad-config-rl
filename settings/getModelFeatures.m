function [model_features] = getModelFeatures(stateActionInfo)
%GETMODELFEATURE Returns the model type and the feature representation for
%the model
%   Includes a mapping between the operator values and the model features
    
    cfg = getConfig();
    J = getJointConfigurationSpace();
    K = size(J,1);
    
    if ~exist('stateActionInfo','var')
        % If not state action info is provided, return 
        % a default vector which can be used to measure the size of the
        % input for the model
        if cfg.model_state_representation == cfg.STATE_IMAGE
            error('please configure here the actual image size')
            %state = zeros(28,28);
        elseif cfg.model_state_representation == cfg.STATE_OPERATOR_POSITION
            state = 1;
        end
        action = J(1,:);
    else
        state = stateActionInfo{1};
        action = stateActionInfo{2};
    end
    
    if cfg.model_state_representation == cfg.STATE_IMAGE
        state_vec = state(:);
    elseif cfg.model_state_representation == cfg.STATE_OPERATOR_POSITION
        
        state_vec = getOneOfKVec(cfg.max_opchain_length, state);
    else
        error('No state representation configured.')
    end
    
    
    if cfg.model_action_representation == cfg.ACTION_JOINT_ONEOFK
        [~,indx]=ismember(J, action, 'rows');
        action_vec = getOneOfKVec(K, find(indx));
    
    elseif cfg.model_action_representation == cfg.ACTION_OPONEOFK_AND_PARAMONEOFK
        numOperators = length(unique(J(:,1)));
        action_vec_p1 = getOneOfKVec(numOperators, action(1));
        
        ct_op_params = J(J(:,1)==action(1),2);
        action_vec_p2 = getOneOfKVec(length(ct_op_params),find(ct_op_params==action(2)));
        action_vec = [action_vec_p1, action_vec_p2];
    
    elseif cfg.model_action_representation == cfg.ACTION_OPONEOFK_AND_PARAMMAPPED
        numOperators = length(unique(J(:,1)));
        action_vec_p1 = getOneOfKVec(numOperators, action(1));
        ct_op_params = J(J(:,1)==action(1),2);
        pos = ct_op_params==action(2);
        lsp = linspace(0,1, length(ct_op_params));
        action_vec_p2 = lsp(pos);
        action_vec = [action_vec_p1, action_vec_p2];
    
    elseif cfg.model_action_representation == cfg.ACTION_OPONEOFK_AND_PARAMRAW
        numOperators = length(unique(J(:,1)));
        action_vec_p1 = getOneOfKVec(numOperators, action(1));
        action_vec_p2 = action(2);
        action_vec = [action_vec_p1, action_vec_p2];
    else
            error('Not yet implemented')
    end
 
    model_features = [state_vec, action_vec];
   
end

function [oneOfKVec] = getOneOfKVec(K, hot_component)
    oneOfKVec = zeros(1,K);
    oneOfKVec(hot_component) = 1;
end

