function [chain_preferences] = generatePairwiseStateOperatorPreferences(image, op_position, qualities, params)
%generatePairwiseStateOperatorPreferences Summary of this function goes here
    
    cfg = params.cfg;
    JointConfigurationSpace = getJointConfigurationSpace();
    
    [~, ordering] = sort(qualities,'descend');
    M = length(ordering);
    chain_pref_count = 1;
    chain_preferences = cell(0,2);
    
    if cfg.sampling_schema == cfg.SAMPLING_PAPI
        i1 = 1;
        winning_action = JointConfigurationSpace(ordering(i1),:);
        for i2 = i1+1 : M
            if (qualities(ordering(i1))-qualities(ordering(i2)) > 0.001) % a > b?, with some eps
                loosing_action = JointConfigurationSpace(ordering(i2),:);
                if cfg.model_state_representation == cfg.STATE_IMAGE
                    if cfg.model_type == cfg.MODEL_PLNET_WITH_CNN_WRAPPER
                        imageFeatures = activations(params.cnn, image, params.cnn_layer, 'OutputAs', 'rows'); % Get Deep CNN FC7 Features
                        chain_preferences{chain_pref_count,1} = {imageFeatures, winning_action}; % winner
                        chain_preferences{chain_pref_count,2} = {imageFeatures, loosing_action}; % looser
                        chain_pref_count = chain_pref_count + 1;
                    else
                        chain_preferences{chain_pref_count,1} = {image(:), winning_action}; % winner
                        chain_preferences{chain_pref_count,2} = {image(:), loosing_action}; % looser
                        chain_pref_count = chain_pref_count + 1;
                    end
                elseif cfg.model_state_representation == cfg.STATE_OPERATOR_POSITION
                    chain_preferences{chain_pref_count,1} = {op_position, winning_action}; % winner
                    chain_preferences{chain_pref_count,2} = {op_position, loosing_action}; % looser
                    chain_pref_count = chain_pref_count + 1;
                end
            end
        end
    elseif cfg.sampling_schema == cfg.SAMPLING_PBPI
        for i1 = 1 : M-1
            winning_action = JointConfigurationSpace(ordering(i1),:);
            for i2 = i1+1 : M
                if (qualities(ordering(i1))-qualities(ordering(i2)) > 0.001) % a > b?, with some eps
                    loosing_action = JointConfigurationSpace(ordering(i2),:);
                    if cfg.model_state_representation == cfg.STATE_IMAGE
                        if cfg.model_type == cfg.MODEL_PLNET_WITH_CNN_WRAPPER
                            imageFeatures = activations(params.cnn, image, params.cnn_layer, 'OutputAs', 'rows'); % Get Deep CNN FC7 Features
                            chain_preferences{chain_pref_count,1} = {imageFeatures, winning_action}; % winner
                            chain_preferences{chain_pref_count,2} = {imageFeatures, loosing_action}; % looser
                            chain_pref_count = chain_pref_count + 1;
                        else
                            chain_preferences{chain_pref_count,1} = {image(:), winning_action}; % winner
                            chain_preferences{chain_pref_count,2} = {image(:), loosing_action}; % looser
                            chain_pref_count = chain_pref_count + 1;
                        end
                    elseif cfg.model_state_representation == cfg.STATE_OPERATOR_POSITION
                        chain_preferences{chain_pref_count,1} = {op_position, winning_action}; % winner
                        chain_preferences{chain_pref_count,2} = {op_position, loosing_action}; % looser
                        chain_pref_count = chain_pref_count + 1;
                    end
                end
            end
        end
    else
        error('No sampling schema configured.')
    end
end

