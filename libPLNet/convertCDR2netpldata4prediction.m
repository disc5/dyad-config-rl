function [ X ] = convertCDR2netpldata4prediction( InstanceFeatures, LabelFeatures )
%CONVERT2NETPLDATA Converts contextual dyad ranking data to the NetPL data
%format.
%
%   NetPL data can be fed into the NetPL method either as training data
%   or as test data. In the latter case the data is used to produce utility
%   scores.
%
%   Inputs:
%       InstanceFeatures - Nxp matrix of instances
%       LabelFeatures    - Mxq matrix of label features
%  
%   Outputs:
%       X - cell array with N observations, where each observation is a
%           Mxt matrix of object vectors (in rows) which is just arranged
%           in the order Label 1, Label 2, ... , Label M.
%
%   Version "4prediction" : this is applicable for an already trained
%   PLNet to obtain skills for each contextual dyad ranking pair (x,L1), (x,L2), ...
%   
%   (C) 2016 Dirk Schäfer 

    numInst = size(InstanceFeatures,1);
    M = size(LabelFeatures,1);
    X = cell(numInst,1);
    for i1 = 1 : numInst
        ctObs = [];
        for i2 = 1 : M
            ctObs = [ctObs; [InstanceFeatures(i1,:),LabelFeatures(i2,:)]];
        end
        X{i1} = ctObs;
    end
end

