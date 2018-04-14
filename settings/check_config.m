% v2: performs v1 multiple times to produce learning curves

%% Preliminaries
clear all;
addpath(genpath('../'))
%%

%% Config
cfg = getConfig();

%%
json=savejson('',cfg,'filename','test.json')
%%
clear all;
%%
data=loadjson('test.json')