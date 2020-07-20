function [ im_flip ] = imflip( im )
%IMFLIP Summary of this function goes here
%   Detailed explanation goes here
        [h,w,c] = size(im);
        tform = maketform('affine',[-1 0 0;0 1 0;w 0 1]);
        im_flip = imtransform(im, tform,'nearest');

end

