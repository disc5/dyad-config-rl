function [chain_preferences] = elicitPipelineOperatorPreferences(policy_model, ct_state0, gt_image)
%EVALUATEOPCHAINPREFERENCES Elicit pairwise (state,op) preferences
%   Generates contextualized preferences by rolling out from different
%   start states.


    cfg = getConfig();
    L = cfg.max_opchain_length;
    JointConfigurationSpace = getJointConfigurationSpace();
    
    chain_preferences = cell(0,2);
    chain_pref_count = 1;
    
    ct_state = ct_state0;
    
    M = size(JointConfigurationSpace,1);
    
    for i1 = 1 : L
       % for all actions on the current state and existing policy...
       ct_qualities = zeros(M,1);
       for i2=1:M
            ct_action = JointConfigurationSpace(i2,:);
            ct_op_id = ct_action(1);
            ct_op_params = ct_action(2:end);
            next_state = applyOperator(ct_op_id, ct_op_params, ct_state);
            [ct_state_end, ~] = rollout(policy_model, next_state, L+1-i1-1, i1); 
            ct_qualities(i2) = calculateImageSimilarity(ct_state_end, gt_image);
       end
       
       % Generate preferences
       [chain_preferences_tmp] = generatePairwiseStateOperatorPreferences(ct_state, i1, ct_qualities);
       
       % Add preferences
       for i5 = 1 : size(chain_preferences_tmp,1)
        chain_preferences{chain_pref_count,1} = chain_preferences_tmp{i5,1};
        chain_preferences{chain_pref_count,2} = chain_preferences_tmp{i5,2};
        chain_pref_count = chain_pref_count + 1;
       end
       
       
       % Continue with the image that was produced with the most promising
       % action
       [~, ordering] = sort(ct_qualities,'descend');
       winning_action = JointConfigurationSpace(ordering(1),:);
       winning_op_id = winning_action(1);
       winning_op_params = winning_action(2:end);
       next_winning_state = applyOperator(winning_op_id, winning_op_params, ct_state);
       
       ct_state = next_winning_state;
    end
end

