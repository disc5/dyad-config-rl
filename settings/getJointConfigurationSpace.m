function [ActionFeatureSpace] = getJointConfigurationSpace()
%GETJOINTCONFIGRATIONSPACE Returns parameters of all used operators
%   Here the discretized parameters of all operators are defined and
%   in a single matrix, called "ActionFeatureSpace"
%
%   The first component of the vector indicates the operator id,
%   whereas the second component corresponds to the operator value.

    %% Joint Configuration Space
    [Op1_Values, Op2_Values, Op3_Values] = getOperatorParameterSpace();
    
    ActionFeatureSpace = zeros(length(Op1_Values)+length(Op2_Values) + 3,2);
    cnt = 1;
    % Op1 : Log
    for i1 = 1:length(Op1_Values)
        ActionFeatureSpace(cnt,:) = [1,Op1_Values(i1)];
        cnt = cnt + 1;
    end
    % Op2: Gamma
    for i1 = 1:length(Op2_Values)
        ActionFeatureSpace(cnt,:) = [2,Op2_Values(i1)];
        cnt = cnt + 1;
    end
    % Op3: Brightness
    for i1 = 1:length(Op3_Values)
        ActionFeatureSpace(cnt,:) = [3,Op3_Values(i1)];
        cnt = cnt + 1;
    end 
    ActionFeatureSpace(cnt,:) = [4,0]; % Unsharping mask
    cnt = cnt + 1;
    ActionFeatureSpace(cnt,:) = [5,0]; % Histogram Equalization
    cnt = cnt + 1;
    ActionFeatureSpace(cnt,:) = [6,0]; % Stop Operator
end

