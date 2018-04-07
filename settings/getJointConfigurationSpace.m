function [ActionFeatureSpace] = getJointConfigurationSpace()
%GETJOINTCONFIGRATIONSPACE Returns parameters of all used operators
%   Here the discretized parameters of all operators are defined and
%   in a single matrix, called "ActionFeatureSpace"
%
%   The first component of the vector indicates the operator id,
%   whereas the second component corresponds to the operator value.

    % Op1 : Log
    % Op2 : Gamma

    %% Joint Configuration Space
    [Op1_Values, Op2_Values] = getOperatorParameterSpace();
    
    ActionFeatureSpace = zeros(length(Op1_Values)+length(Op2_Values),2);
    cnt = 1;
    for i1 = 1:length(Op1_Values)
        ActionFeatureSpace(cnt,:) = [1,Op1_Values(i1)];
        cnt = cnt + 1;
    end
    for i1 = 1:length(Op2_Values)
        ActionFeatureSpace(cnt,:) = [2,Op2_Values(i1)];
        cnt = cnt + 1;
    end 
end

