function showDistRestoredOriginals(originals, distorted, restored)
%SHOWDISTRESTOREDORIGINALS Summary of this function goes here
    %% Inspection
    nImages = size(originals,1);

    showNumImages = 25;
    if nImages < showNumImages
        error('Not enough images.')
    end
    
     % Distorted
    figure;
    for i1=1:showNumImages
        subplot(5,5,i1)
        imshow(distorted{i1})
    end
    currentFigure = gcf;
    title(currentFigure.Children(end), 'Inputs (Distorted Images)');

     % Restored
    figure;
    for i1=1:showNumImages
        subplot(5,5,i1)
        imshow(restored{i1})
    end
    currentFigure = gcf;
    title(currentFigure.Children(end), 'Outputs (Enhanced Images)');
    
    % Originals
    figure;
    for i1=1:showNumImages
        subplot(5,5,i1)
        imshow(originals{i1})
    end
    currentFigure = gcf;
    title(currentFigure.Children(end), 'Originals');

   
   
end

