function [op_stats,action_winner_stats,action_looser_stats ] = analyse_pw_prefs(ct_chain_preferences,current_op_slot)
%ANALYSE_PW_PREFS Summary of this function goes here
%   Detailed explanation goes here
    J = getJointConfigurationSpace();
    i1 = current_op_slot; % op_slot 1
    op_stats = zeros(4,1);
    action_winner_stats = zeros(4,length(getJointConfigurationSpace));
    action_looser_stats = zeros(4,length(getJointConfigurationSpace));
    for i2 = 1:length(ct_chain_preferences)
        if ct_chain_preferences{i2,1}{1} == 1
            op_stats(i1) = op_stats(i1) + 1;
            [~,idx]=ismember(ct_chain_preferences{i2,1}{2},J,'rows');
            action_winner_stats(i1, idx) = action_winner_stats(i1, idx) + 1;
            [~,idx]=ismember(ct_chain_preferences{i2,2}{2},J,'rows');
            action_looser_stats(i1, idx) = action_looser_stats(i1, idx) + 1;
        end
    end
end

