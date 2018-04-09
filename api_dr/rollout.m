function [state_end, seq_choices] = rollout(policy_model, state_start, K, current_op_pos, round)
%ROLLOUT Performs a rollout using a policy model
%   The function applies the policy model along the different stages of the pipeline.
%
%   Input:
%       policy_model - E.g., PLNet
%       state_start - start input image
%       K - trajectory length of the rollout
%       round - the round number, this info can be used to implement a
%       cool-down schedule for exploration
%   
%   Output:
%       state_end - resulting image
%
%   (C) 2018 Dirk Schaefer

    cfg = getConfig();
    
    JointConfigurationSpace = getJointConfigurationSpace();
    seq_choices = cell(K,1);
    ct_state = state_start;
    
    boltzmann_schedule = [0.01,0.05,0.1,0.5,1,10,100,100];
    
    for i2 = 1 : K
        [ordering, skills] = getActionRankingGivenState(policy_model, current_op_pos);
        if cfg.boltzman_exploration == true
            skills2 = log(skills);
            c = boltzmann_schedule(round);
            for i8 = 1 : length(skills2)
                skills2(i8) = exp(c*skills2(i8));
            end
            prob = skills2./sum(skills2);
            try
                [foo,idx] = histc(rand(1,1),[0,cumsum(prob')]); % sample from the prob distribution
            catch ME
                printf('Exception caught!');
            end
            ct_action = JointConfigurationSpace(idx,:);
        else
            ct_action = JointConfigurationSpace(ordering(1),:); % take the best action
        end
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

