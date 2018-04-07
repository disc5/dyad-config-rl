function [state_end, seq_choices] = rollout(policy_model, state_start, K, current_op_pos)
%ROLLOUT Performs a rollout using a policy model
%   The function applies the policy model along the different stages of the pipeline.
%
%   Input:
%       policy_model - E.g., PLNet
%       state_start - start input image
%       K - trajectory length of the rollout
%   
%   Output:
%       state_end - resulting image
%
%   (C) 2018 Dirk Schaefer

    JointConfigurationSpace = getJointConfigurationSpace();
    seq_choices = cell(K,1);
    ct_state = state_start;
    
    for i2 = 1 : K
        [ordering, ~] = getActionRankingGivenState(policy_model, current_op_pos);

        ct_action = JointConfigurationSpace(ordering(1),:);
        ct_op_id = ct_action(1);
        ct_op_params = ct_action(2:end);
        
        seq_choices{i2} = {ct_state, ct_action};
        
        ct_state = applyOperator(ct_op_id, ct_op_params, ct_state);
       
        if (sum(isnan(ct_state)) > 0)
                fprintf('Fatal error: produced nan value!');
        end
    end
    state_end = ct_state;
end

