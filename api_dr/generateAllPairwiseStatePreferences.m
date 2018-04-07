function [chain_preferences] = generateAllPairwiseStatePreferences(state, winning_op_id, winning_op_param, op_position)
%GENERATEALLPAIRWISESTATEPREFERENCES Summary of this function goes here
    
    [ cfg ] = getConfig();

    [Op1_Values, Op2_Values] = getOperatorParameterSpace();
    chain_pref_count = 1;
    
    if cfg.model_state_representation == cfg.STATE_OPERATOR_POSITION
        if winning_op_id == 1
         for i2 = 1 : length(Op1_Values)
             if Op1_Values(i2)~=winning_op_param
                chain_preferences{chain_pref_count,1} = {[winning_op_id, winning_op_param, op_position]}; % winner
                chain_preferences{chain_pref_count,2} = {[1, Op1_Values(i2), op_position]}; % looser
                chain_pref_count = chain_pref_count + 1;
             end
         end
         for i2 = 1 : length(Op2_Values)
                chain_preferences{chain_pref_count,1} = {[winning_op_id, winning_op_param, op_position]}; % winner
                chain_preferences{chain_pref_count,2} = {[2, Op2_Values(i2), op_position]}; % looser
                chain_pref_count = chain_pref_count + 1;
         end
         
        else
            for i2 = 1 : length(Op1_Values)
                    chain_preferences{chain_pref_count,1} = {[winning_op_id, winning_op_param, op_position]}; % winner
                    chain_preferences{chain_pref_count,2} = {[1, Op1_Values(i2), op_position]}; % looser
                    chain_pref_count = chain_pref_count + 1;

             end
             for i2 = 1 : length(Op2_Values)
                 if Op2_Values(i2)~=winning_op_param
                    chain_preferences{chain_pref_count,1} = {[winning_op_id, winning_op_param, op_position]}; % winner
                    chain_preferences{chain_pref_count,2} = {[2, Op2_Values(i2), op_position]}; % looser
                    chain_pref_count = chain_pref_count + 1;
                 end
             end
        end
    elseif cfg.model_state_representation == cfg.STATE_IMAGE
         if winning_op_id == 1
             for i2 = 1 : length(Op1_Values)
                 if Op1_Values(i2)~=winning_op_param
                    chain_preferences{chain_pref_count,1} = {state(:), [winning_op_id, winning_op_param, op_position]}; % winner
                    chain_preferences{chain_pref_count,2} = {state(:), [1, Op1_Values(i2), op_position]}; % looser
                    chain_pref_count = chain_pref_count + 1;
                 end
             end
             for i2 = 1 : length(Op2_Values)
                    chain_preferences{chain_pref_count,1} = {state(:), [winning_op_id, winning_op_param, op_position]}; % winner
                    chain_preferences{chain_pref_count,2} = {state(:), [2, Op2_Values(i2), op_position]}; % looser
                    chain_pref_count = chain_pref_count + 1;
             end

        else
            for i2 = 1 : length(Op1_Values)
                    chain_preferences{chain_pref_count,1} = {state(:), [winning_op_id, winning_op_param, op_position]}; % winner
                    chain_preferences{chain_pref_count,2} = {state(:), [1, Op1_Values(i2), op_position]}; % looser
                    chain_pref_count = chain_pref_count + 1;

             end
             for i2 = 1 : length(Op2_Values)
                 if Op2_Values(i2)~=winning_op_param
                    chain_preferences{chain_pref_count,1} = {state(:), [winning_op_id, winning_op_param, op_position]}; % winner
                    chain_preferences{chain_pref_count,2} = {state(:), [2, Op2_Values(i2), op_position]}; % looser
                    chain_pref_count = chain_pref_count + 1;
                 end
             end
        end
    else
        error('To be implemented')
    end
end

