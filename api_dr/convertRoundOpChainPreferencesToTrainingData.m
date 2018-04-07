function [trainingData] = convertRoundOpChainPreferencesToTrainingData(round_chain_comparisons)
%convertRoundOpChainPreferencesToTrainingData Converts chain preferences into actual training data
%   The functions creates training data which format highly depend on the
%   model type. 
%
%   Input:
%       chain_comparison - L x K^2 x 2 cell array of (state,action) choices,
%           where the choice at the first column is preferred over that of
%           the second column, L of such Op chain comparisons were
%           generated within a round.
%
%   Output:
%       trainingData - cell array of training examples
%
% (C) 2018 Dirk Schaefer
    cfg = getConfig();

    L = size(round_chain_comparisons,1);    
    trainingData = cell(0);
    cnt = 1;
    for i1 = 1 : L
        [roundPrefs] = convertSingleOpChainPreferencesToTrainingData(round_chain_comparisons{i1});
        if cfg.model_type == cfg.MODEL_PLNET
            for i2 = 1 : size(roundPrefs,1)
                trainingData{cnt} = squeeze(roundPrefs(i2,:,:));
                cnt = cnt + 1;
            end
        else
            error('To be implemented')
        end
    end
    trainingData = trainingData';
end

