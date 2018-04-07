%% Test Ground on Data Generation
% - Is it possible to invert the data?
% - Which quality measure is suitable?
% -- It should be chosen s.t. the final distorted image has lower quality as the 
%    distorted intermediate images.

clear all
%%
addpath('../ext/')
addpath('../image_operators/')
addpath('../helper/')
%%

% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
images = loadMNISTImages('../fashionMNIST/train-images-idx3-ubyte');
labels = loadMNISTLabels('../fashionMNIST/train-labels-idx1-ubyte');

%% -------------------------------------
%% Image selection
nImages = 25;
originals=cell(nImages,1);
for i1=1:nImages
    id = i1;
    img = images(:,id); % values between 0 and 1
    I=reshape(img,28,28);
    originals{i1} = I;
end
%% Test
%imshow(originals{12})

%% Preliminaries
Op_log_values = [1.5, 2, 2.5, 3, 3.5, 4]; % 0.5, 1
Op_gamma_values = [0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0];

%%
% Preliminary: randomly ordered chains with Ops 1-3 (Log, Bright, Gamma)
% Result: the images are not inverted properly
% Need to find the conditions when they can be inverted
%  or need to find the reason why they can not be inverted!

% ...
% Conclusion: I'll exclude the brightness operator, because of the loss of
% information is causes.

%% Simplified chain: 1 filter - log
distorted=cell(nImages,1);
restored=cell(nImages,1);
perms = cell(nImages,1);
params = cell(nImages,3);
for i1=1:nImages
    rpi = randperm(1);
    perms{i1} = rpi;
    
    id = i1;
    img = images(:,id);     % values between 0 and 1
    I=reshape(img,28,28);
    
    I_current = I;
    for i2 = 1 : length(rpi)
        op_id = rpi(i2);
        if op_id == 1 % Log
            rparam_val = Op_log_values(randi(length(Op_log_values)));
            params{i1,i2} = rparam_val;
            [I_current] = applyInvOperator(op_id, rparam_val, I);
        end
    end
    restored{i1} = applyOperator(rpi(1), params{i1,1}, I_current);
          
    distorted{i1} = I_current;
end

%%
%% Inspection
showDistRestoredOriginals(originals, distorted, restored)
% => Okay

%% Similarity Inspection / Check for invertability
avgsim = 0;
for i1 = 1:nImages
    sim = calculateImageSimilarity( restored{i1},originals{i1} );
    fprintf('Example %d - Sim = %3.4f \n', i1, sim);
    avgsim = avgsim+sim;
end
avgsim = avgsim/nImages;
fprintf('Avg Sim = %3.4f \n', avgsim);


%% Simplified chain: 1 filter - gamma (=:Op2)
distorted=cell(nImages,1);
restored=cell(nImages,1);
params = cell(nImages,1);
for i1=1:nImages
    
    id = i1;
    img = images(:,id);     % values between 0 and 1
    I=reshape(img,28,28);
    
    I_current = I;

    rparam_val = Op_gamma_values(randi(length(Op_gamma_values)));
    params{i1,1} = rparam_val;
    [I_current] = applyInvOperator(2, rparam_val, I_current);
   

    restored{i1} = applyOperator(2, params{i1,1}, I_current);
          
    distorted{i1} = I_current;
end

%%
%% Inspection
showDistRestoredOriginals(originals, distorted, restored)
% -> Okay

%% Similarity Inspection / Check for invertability
avgsim = 0;
for i1 = 1:nImages
    sim = calculateImageSimilarity( restored{i1},originals{i1} );
    fprintf('Example %d - Sim = %3.4f \n', i1, sim);
    avgsim = avgsim+sim;
end
avgsim = avgsim/nImages;
fprintf('Avg Sim = %3.4f \n', avgsim);


%% Construction of a simple chain using 2 ops
%i.e. Distorted -> Op1(log) -> Op2 (gamma) -> Restore
% Building the inverse by 
% Original -> Op2^(-1)(gamma) -> Op1^(-1)(log)->Distorted

distorted=cell(nImages,1);
restored=cell(nImages,1);
%op_seq = cell(nImages,1); fixed
params = cell(nImages,2);
for i1=1:nImages
    id = i1;
    img = images(:,id);     % values between 0 and 1
    I=reshape(img,28,28);
    
    I_current = I;

    rparam_val = Op_gamma_values(randi(length(Op_gamma_values)));
    params{i1,2} = rparam_val;
    [I_current] = applyInvOperator(2, rparam_val, I_current);

    rparam_val = Op_log_values(randi(length(Op_log_values)));
    params{i1,1} = rparam_val;
    [I_current] = applyInvOperator(1, rparam_val, I_current);
   
    % Forward application, aim to restore original
    I_FStep1 = applyOperator(1, params{i1,1}, I_current);
    I_FStep2 = applyOperator(2, params{i1,2}, I_FStep1);
    
    restored{i1} = I_FStep2;
    distorted{i1} = I_current;
end

%%
showDistRestoredOriginals(originals, distorted, restored)
% => Okay

%% Similarity Inspection / Check for invertability
avgsim = 0;
for i1 = 1:nImages
    sim = calculateImageSimilarity( restored{i1},originals{i1} );
    fprintf('Example %d - Sim = %3.4f \n', i1, sim);
    avgsim = avgsim+sim;
end
avgsim = avgsim/nImages;
fprintf('Avg Sim = %3.4f \n', avgsim);


%% Construction of more complicated chains using these two operators (with fixed order)
% i.e., Op1 (log) and Op2 (gamma)
% Op1->Op1->Op2->Op1 with random parameter values
% thus the inverse sequence is Op1<-Op2<-Op1<-Op1

% Neutral param for Op2 (gamma) is c=0
% Neutral param for Op1 (log) unknown
%
% Proof of concept of chain irreversibility.
distorted=cell(nImages,1);
restored=cell(nImages,1);
%op_seq = cell(nImages,1); fixed
params = cell(nImages,4);
for i1=1:nImages
    id = i1;
    img = images(:,id);     % values between 0 and 1
    I=reshape(img,28,28);
    
    I_current = I;

    rparam_val = Op_log_values(randi(length(Op_log_values)));
    params{i1,4} = rparam_val;
    [I_current] = applyInvOperator(1, rparam_val, I_current);
   
    rparam_val = Op_gamma_values(randi(length(Op_gamma_values)));
    params{i1,3} = rparam_val;
    [I_current] = applyInvOperator(2, rparam_val, I_current);

    rparam_val = Op_log_values(randi(length(Op_log_values)));
    params{i1,2} = rparam_val;
    [I_current] = applyInvOperator(1, rparam_val, I_current);
    
    rparam_val = Op_log_values(randi(length(Op_log_values)));
    params{i1,1} = rparam_val;
    [I_current] = applyInvOperator(1, rparam_val, I_current);

    % Forward application, aim to restore original
    I_FStep1 = applyOperator(1, params{i1,1}, I_current);
    I_FStep2 = applyOperator(1, params{i1,2}, I_FStep1);
    I_FStep3 = applyOperator(2, params{i1,3}, I_FStep2);
    I_FStep4 = applyOperator(1, params{i1,4}, I_FStep3);
    
    restored{i1} = I_FStep4;
          
    distorted{i1} = I_current;
end

%% Inspection
showDistRestoredOriginals(originals, distorted, restored)
% => Not quite okay, there can be loss of information
% Observed when last (forward) op has param 0.5 or 1
%% Similarity Inspection / Check for invertability
avgsim = 0;
for i1 = 1:nImages
    sim = calculateImageSimilarity( restored{i1},originals{i1} );
    fprintf('Example %d - Sim = %3.4f \n', i1, sim);
    avgsim = avgsim+sim;
end
avgsim = avgsim/nImages;
fprintf('Avg Sim = %3.4f \n', avgsim);

%% Similarities between distorted and originals
avgsim = 0;
for i1 = 1:nImages
    sim = calculateImageSimilarity( distorted{i1},originals{i1} );
    fprintf('Example %d - Sim = %3.4f \n', i1, sim);
    avgsim = avgsim+sim;
end
avgsim = avgsim/nImages;
fprintf('Avg Sim = %3.4f \n', avgsim);



