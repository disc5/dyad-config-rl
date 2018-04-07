function [processedImage, op_seq, op_seq_ids, intermediates] = applyPolicy(policy_model, image)
%APPLYPOLICY Applies a learned policy a fixed amount of steps
%   This function can be used for evaluation.
%
%   Input:
%       policy_model - joint-feature PL model
%       image - a distorted image
%
%   Output:
%       processedImage - the outcome
%
% (C) 2018 Dirk Schaefer
    
    [ cfg ] = getConfig();
    max_opchain_length = cfg.max_opchain_length;
    JointConfigurationSpace = getJointConfigurationSpace();
    
    intermediates = cell(max_opchain_length+1, 1);
    intermediates{1}=image;
    
    ct_state = image;
    op_seq = cell(max_opchain_length,1);
    op_seq_ids = cell(max_opchain_length,1);
    
    for i2 = 1:max_opchain_length
        [ordering, ~] = getActionRankingGivenState(policy_model, i2);
        ct_action = JointConfigurationSpace(ordering(1),:);
        ct_op_id = ct_action(1);
        ct_op_params = ct_action(2);
        
        op_seq_ids{i2} = ct_op_id; 
        op_seq{i2} = ct_op_params;
        
        ct_state = applyOperator(ct_op_id, ct_op_params, ct_state);
        
        intermediates{i2+1} = ct_state;
    end
    
    processedImage = ct_state;
end

