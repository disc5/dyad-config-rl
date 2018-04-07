%%
clear all
addpath(genpath('/Users/dschaefer/Dropbox/code/mylibs'))
%%
addpath('ext/')
addpath('image_operators/')

%%
images = loadMNISTImages('fashionMNIST/train-images-idx3-ubyte');
labels = loadMNISTLabels('fashionMNIST/train-labels-idx1-ubyte');
 
%% Display an image
id = 1;
img = images(:,id); % values between 0 and 1
I=reshape(img,28,28);
imshow(I)

%% -----------------------------------
%% FILTER DEFINITIONS:
%% -----------------------------------
%% Filter 1: Logarithm Operator
c=1.5; % value range: 0.5, 1, 1.5 ... 4.5, 5
[I2] = logarithmicOperator(I,c)
subplot(1,2,1)
imshow(I)
subplot(1,2,2)
imshow(I2)
%ssim(I2,I)
FeatureSIM(I2,I)

%% -----------------------------------
%% Filter 2: Brightness
c = -0.2; % value range: -0.3 .. 0.3
[I2] = adjustBrightness(I, c);
subplot(1,2,1)
imshow(I)
subplot(1,2,2)
imshow(I2)
ssim(I2,I)
FeatureSIM(I2,I)

%% -----------------------------------
%% Filter 3: Power law transform (postponed, it has 2 parameters)
% c=0.5;
% gamma = 1.5;
% [I2] = powerlawTransform(I,c, gamma)
% subplot(1,2,1)
% imshow(I)
% subplot(1,2,2)
% imshow(I2)
% FeatureSIM(I2,I)

%% -----------------------------------
%% On reversing the effect of Ops:
%% -----------------------------------
%% Reverse Logarithmic Op (F1)
c = 2.5;
I1 = logarithmicOperator(I,c)
[I2] = invLogarithmicOperator(I1, c)
FeatureSIM(I2,I)

%% Reverse Brightness Op (F2)
c = -0.3;
[I1] = adjustBrightness(I, c);
[I2] = invAdjustBrightness(I1, c)

subplot(1,3,1)
imshow(I)
subplot(1,3,2)
imshow(I1)
subplot(1,3,3)
imshow(I2)
ssim(I2,I)
FeatureSIM(I2,I)

%% Verkettung zweier Operatoren:
% Distorted Image -> OP1 -> OP2 -> Result(Resamples Original)
% Where OP1 = Logarithmic Op, OP2 = Brightness

%% Constrction of the distorted image:
% Original -> InvBright -> InvLog -> Distorted
cLog = 2.5;
cBright = 0.3; % Note for negative c,e.g. -0.3 is may be better to use ssim;

[I_temp] = invAdjustBrightness(I, cBright)
[IDistorted] = invLogarithmicOperator(I_temp, cLog)


%% Proof: Applying operator chain on the Distorted Image
I_temp2 = logarithmicOperator(IDistorted,cLog);
FeatureSIM(I_temp2,I_temp)
I_Result = adjustBrightness(I_temp2, cBright);

%%
subplot(1,3,1)
imshow(IDistorted);
title('Distorted'); % Place title here

subplot(1,3,2)
imshow(I_Result)
title('Result');

subplot(1,3,3)
imshow(I)
title('Original')

%%
FeatureSIM(IDistorted,I)
FeatureSIM(I_Result,I)

%%
ssim(IDistorted,I)
ssim(I_Result,I)



%% -------------------------------------
%% Data set inspection: first 100 images
%% -------------------------------------
close all;
for i1=1:100
    id = i1;
    img = images(:,id); % values between 0 and 1
    I=reshape(img,28,28);
    subplot(10,10,i1)
    imshow(I)
end