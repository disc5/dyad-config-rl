function [state_end, seq_choices] = rollout(policy_model, state_start, K, calling_op_pos, round)
%ROLLOUT Performs a rollout using a policy model
%   The function applies the policy model along the different stages of the pipeline.
%
%   Input:
%       policy_model - E.g., PLNet
%       state_start - start input image
%       K - trajectory length of the rollout
%       calling_op_pos - the op slot from which the rollout is started
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
    
    boltzmann_schedule = cfg.boltzmann_schedule;
    
    for i2 = 1 : K
        [ordering, skills] = getActionRankingGivenState(policy_model, calling_op_pos + i2);
        if cfg.boltzmann_exploration == true
            skills2 = log(skills);
            if round <= length(boltzmann_schedule)
                c = boltzmann_schedule(round);
            else
                c = max(boltzmann_schedule);
            end
            %fprintf('Boltzmann exploration constant in round %d: %3.4f\n', round, c);
            for i8 = 1 : length(skills2)
                skills2(i8) = exp(c*skills2(i8));
            end
            prob = skills2./sum(skills2);
            try
                [~,idx] = histc(rand(1,1),[0,cumsum(prob')]); % sample from the probability distribution
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

