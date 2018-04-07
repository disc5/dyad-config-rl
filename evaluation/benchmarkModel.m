function [result] = benchmarkModel(policy_model, I_Distorted, I_Original)
%BENCHMARKMODEL Benchmark PLNet Operator Configuration Model
%   Provides all intermediate results
%
%   Input:
%       policy_model - PLNet model
%       I_Distorted - the input image
%       I_Original - the groundtruth image
%
%   Output:
%       result - struct with following fields
%           params - cell array, all chosen /predicted operator parameter configurations
%           images - cell arry, all images including those that were produces as a result of an operator
%               application
%           similarities - cell array, all comparisons against groundtruth img
%
% (C) 2018 Dirk Schaefer

JointConfigurationSpace = getJointConfigurationSpace();


% Processing
[ordering, ~] = getRestrictedActionRankingGivenState_PLNet(policy_model, I_Distorted, 1);
ct_action = JointConfigurationSpace(ordering(1),:);
op1_params = ct_action(2);
I_Op1_Result = logarithmicOperator(I_Distorted, op1_params);

[ordering, ~] = getRestrictedActionRankingGivenState_PLNet(policy_model, I_Op1_Result, 2);
ct_action = JointConfigurationSpace(ordering(1),:);
op2_params = ct_action(2);
I_Op2_Result = adjustBrightness(I_Op1_Result, op2_params); 


result.images{1} = I_Distorted;
result.images{2} = I_Op1_Result;
result.images{3} = I_Op2_Result;
result.images{4} = I_Original;

result.params{1} = op1_params;
result.params{2} = op2_params;

result.similarities{1} = calculateImageSimilarity(I_Distorted,I_Original);
result.similarities{2} = calculateImageSimilarity(I_Op1_Result,I_Original);
result.similarities{3} = calculateImageSimilarity(I_Op2_Result,I_Original);

end

