%% Preliminaries
clear all;
addpath(genpath('../../'))

%%
load('../e1_variable_pipeline_lengths_v2.mat');

%%
params.empty = 1;
[total_error_te, te_restored,allIntermediates] = evaluatePolicy(best_policy_model, te_distorted, te_originals, params);
total_error_te
%%
showDistRestoredOriginals(te_originals, te_distorted, te_restored)

%%
IDS = [21,1,4,14];

%%
Inter = allIntermediates{21}

%%
close all
currentID = 4;
[total_error_te, te_restored,allIntermediates] = evaluatePolicy(best_policy_model, {te_distorted{currentID}}, {te_originals{currentID}}, params);
for i2 = 1 : 5
        I = allIntermediates{1}{i2};
        subplot(1,6,i2)
        imshow(I)
end
subplot(1,6,6)
imshow(te_originals{currentID});

%%
