%% Data set Generator
% 
% This script produces distorted images by applying a data pipeline in the
% opposite direction.
%
% full-reference scenario, i.e. the quality of a pipeline can be measured
% against a ground-truth
%
clear all
%%
addpath(genpath('../'))
%%

% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
images = loadMNISTImages('../fashionMNIST/train-images-idx3-ubyte');
labels = loadMNISTLabels('../fashionMNIST/train-labels-idx1-ubyte');

%% Load valid operator ranges
[Op_log_values, Op_gamma_values] = getOperatorParameterSpace()

% Fix some params
Op_log1 = 2.5;
Op_log2 = 1.5;
Op_log3 = 2.0;
Op_gamma1 = 1.4;
%% -------------------------------------
%% Data set inspection: first 100 images
%% -------------------------------------
close all;
%%
% figure
% for i1=1:100
%     id = i1;
%     img = images(:,id); % values between 0 and 1
%     I=reshape(img,28,28);
%     subplot(10,10,i1)
%     imshow(I)
% end

%% -------------------------------------
%% Data set generation on the first 100 images
%% -------------------------------------
nTrImages = 500;
originals=cell(nTrImages,1);
for i1=1:nTrImages
    id = i1;
    img = images(:,id); % values between 0 and 1
    I=reshape(img,28,28);
    originals{i1} = I;
end

%% Construction of  chains using two base operators (with fixed order)
% i.e., Op1 (log) and Op2 (gamma)
% Op1->Op1->Op2->Op1 with random parameter values
% thus the inverse sequence is Op1<-Op2<-Op1<-Op1
%
distorted=cell(nTrImages,1);
params = cell(nTrImages,4);
for i1=1:nTrImages
    id = i1;
    img = images(:,id);     % values between 0 and 1
    I=reshape(img,28,28);
    
    I_current = I;

    %rparam_val = Op_log_values(randi(length(Op_log_values)));
    params{i1,4} = Op_log1;
    [I_current] = applyInvOperator(1, Op_log1, I_current);
   
    %rparam_val = Op_gamma_values(randi(length(Op_gamma_values)));
    params{i1,3} = Op_gamma1;
    [I_current] = applyInvOperator(2, Op_gamma1, I_current);

    %rparam_val = Op_log_values(randi(length(Op_log_values)));
    params{i1,2} = Op_log2;
    [I_current] = applyInvOperator(1, Op_log2, I_current);
    
    %rparam_val = Op_log_values(randi(length(Op_log_values)));
    params{i1,1} = Op_log3;
    [I_current] = applyInvOperator(1, Op_log3, I_current);
          
    distorted{i1} = I_current;
end


%% Save
save('../data/fashion-mnist-distort100-4ch.mat','originals','distorted')


%% =======================================================================
%% Generate TEST Data
% LOADs
te_images = loadMNISTImages('../fashionMNIST/t10k-images-idx3-ubyte');
te_labels = loadMNISTLabels('../fashionMNIST/t10k-labels-idx1-ubyte');

%% -------------------------------------
%% Data set generation on the first 100 images
%% -------------------------------------
nTeImages = 50;
te_originals=cell(nTeImages,1);
for i1=1:nTeImages
    id = i1;
    img = te_images(:,id); % values between 0 and 1
    I=reshape(img,28,28);
    te_originals{i1} = I;
end

%%
%% Construction of  chains using two base operators (with fixed order)
% i.e., Op1 (log) and Op2 (gamma)
% Op1->Op1->Op2->Op1 with random parameter values
% thus the inverse sequence is Op1<-Op2<-Op1<-Op1
%
te_distorted=cell(nTeImages,1);
params = cell(nTeImages,4);
for i1=1:nTeImages
    id = i1;
    img = te_images(:,id);     % values between 0 and 1
    I=reshape(img,28,28);
    
    I_current = I;

     %rparam_val = Op_log_values(randi(length(Op_log_values)));
    params{i1,4} = Op_log1;
    [I_current] = applyInvOperator(1, Op_log1, I_current);
   
    %rparam_val = Op_gamma_values(randi(length(Op_gamma_values)));
    params{i1,3} = Op_gamma1;
    [I_current] = applyInvOperator(2, Op_gamma1, I_current);

    %rparam_val = Op_log_values(randi(length(Op_log_values)));
    params{i1,2} = Op_log2;
    [I_current] = applyInvOperator(1, Op_log2, I_current);
    
    %rparam_val = Op_log_values(randi(length(Op_log_values)));
    params{i1,1} = Op_log3;
    [I_current] = applyInvOperator(1, Op_log3, I_current);
          
    te_distorted{i1} = I_current;
end


%%
save('../data/fashion-mnist-test-distort100-4ch.mat','te_originals','te_distorted')
