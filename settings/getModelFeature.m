function [model_feature] = getModelFeature(actionFeatureVector)
%GETMODELFEATURE Returns the model type and the feature representation for
%the model
%   Includes a mapping between the operator values and the model features
    op_id = actionFeatureVector(1);
    
    JointConfigurationSpace = getJointConfigurationSpace();
    
    K = size(JointConfigurationSpace,1);
    
    if op_id == 1
        %% Op1: Log Op
        %Op1_ModelFeatureSpace = linspace(0,1,length(Op1_Values));
%         Op1_ModelFeatureSpace = linspace(-1,1,length(Op1_Values));
%         Op1_Value2FeatureMapping = ones(length(Op1_Values),2);
%         Op1_Value2FeatureMapping(:,1) = Op1_Values';
%         Op1_Value2FeatureMapping(:,2) = Op1_ModelFeatureSpace';
%         
%         for i1 = 1 : size(Op1_Value2FeatureMapping,1)
%             if (Op1_Value2FeatureMapping(i1,1) == actionFeatureVector(2))
%                  model_feature = [1,0, Op1_Value2FeatureMapping(i1,2)];
%                  break;
%             end
%         end

        for i1 = 1 : K
            if (JointConfigurationSpace(i1,1) == op_id && JointConfigurationSpace(i1,2) == actionFeatureVector(2))
                model_feature = zeros(1,K+4); % 1-of-K encoding
                model_feature(i1) = 1;
                model_feature(K+actionFeatureVector(3))=1;
                break;
            end
        end
    
    else 

        %% Op2: Gamma
        %Op2_ModelFeatureSpace = linspace(0,1,length(Op2_Values));
%         Op2_ModelFeatureSpace = linspace(-1,1,length(Op2_Values));
%         Op2_Value2FeatureMapping = ones(length(Op2_Values),2);
%         Op2_Value2FeatureMapping(:,1) = Op2_Values';
%         Op2_Value2FeatureMapping(:,2) = Op2_ModelFeatureSpace';
%         
%         for i1 = 1 : size(Op2_Value2FeatureMapping,1)
%             if (Op2_Value2FeatureMapping(i1,1) == actionFeatureVector(2))
%                  model_feature = [0,1, Op2_Value2FeatureMapping(i1,2)];
%                  break;
%             end
%         end
        for i1 = 1 : K
            if (JointConfigurationSpace(i1,1) == op_id && JointConfigurationSpace(i1,2) == actionFeatureVector(2))
                model_feature = zeros(1,K+4); % 1-of-K encoding
                model_feature(i1) = 1;
                model_feature(K+actionFeatureVector(3))=1;
                break;
            end
        end
        
    end

   
end

