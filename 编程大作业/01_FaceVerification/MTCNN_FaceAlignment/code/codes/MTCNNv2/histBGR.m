function [ output ] = histBGR( img)
    output = zeros(size(img));
    output(:,:,1) = histeq(img(:,:,1));
    output(:,:,2) = histeq(img(:,:,2));
    output(:,:,3) = histeq(img(:,:,3));
    output = uint8(output);
    
end