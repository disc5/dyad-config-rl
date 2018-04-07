function showDistRestoredOriginals(originals, distorted, restored)
%SHOWDISTRESTOREDORIGINALS Summary of this function goes here
%   Detailed explanation goes here
%% Inspection
nImages = 25;%size(originals,1);

% Originals
figure;
for i1=1:nImages
    subplot(5,5,i1)
    imshow(originals{i1})
end

% Distorted
figure;
for i1=1:nImages
    subplot(5,5,i1)
    imshow(distorted{i1})
end

% Restored
figure;
for i1=1:nImages
    subplot(5,5,i1)
    imshow(restored{i1})
end
end

