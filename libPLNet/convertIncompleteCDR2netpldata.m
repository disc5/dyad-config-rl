function [ X ] = convertIncompleteCDR2netpldata( InstanceFeatures, LabelFeatures, LabelOrderings )
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
%       LabelOrderings   - NxM matrix of orderings organised in rows
%  
%   Outputs:
%       X - cell array with N observations, where each observation is a
%           Mxt matrix of object vectors (in rows) which a arranged
%           according the LabelOrderings described above.
%
%   (C) 2016 Dirk Schäfer 

    numInst = size(InstanceFeatures,1);
    M = size(LabelFeatures,1);
    X = cell(numInst,1);
    for i1 = 1 : numInst
        ctObs = [];
        Mn = length(LabelOrderings(i1,LabelOrderings(i1,:)~=-1));
        for i2 = 1 : Mn
            ctObs = [ctObs; [InstanceFeatures(i1,:),LabelFeatures(LabelOrderings(i1,i2),:)]];
        end
        X{i1} = ctObs;
    end
end

