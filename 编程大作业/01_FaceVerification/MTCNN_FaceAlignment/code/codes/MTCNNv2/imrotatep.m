    function B=imrotatep(A,ANGLE,METHOD,BBOX,POINT)
    %IMROTATE Rotate image.
    % B = IMROTATE(A,ANGLE,METHOD) rotates the image A by ANGLE
    % degrees in a counter-clockwise direction, using the specified
    % interpolation method. METHOD is a string that can have one of
    % these values:
    %
    % 'nearest' (default) nearest neighbor interpolation
    %
    % 'bilinear' bilinear interpolation
    %
    % 'bicubic' bicubic interpolation
    %
    % If you omit the METHOD argument, IMROTATE uses the default
    % method of 'nearest'. To rotate the image clockwise, specify a
    % negative angle.
    %
    % B = IMROTATE(A,ANGLE,METHOD,BBOX) rotates the image A through ANGLE
    % degrees. The bounding box of the image is set by the BBOX argument, a
    % string that can be 'loose' (default) or 'crop'. When BBOX is 'loose', B
    % includes the whole rotated image, which generally is larger than A. When
    % BBOX is 'crop' B is cropped to include only the central portion of the
    % rotated image and is the same size as A. If you omit the BBOX argument,
    % IMROTATE uses the default 'loose' bounding box.
    %
    % IMROTATE sets invalid values on the periphery of B to 0.
    %
    % Class Support
    % -------------
    % The input image can be numeric or logical. The output image is of the
    % same class as the input image.
    %
    % Example
    % -------
    % I = fitsread('solarspectra.fts');
    % I = mat2gray(I);
    % J = imrotate(I,-1,'bilinear','crop');
    % imview(I), imview(J)
    %
    % See also IMCROP, IMRESIZE, IMTRANSFORM, TFORMARRAY.

    % Copyright 1993-2003 The MathWorks, Inc.
    % $Revision: 5.25.4.5 $ $Date: 2003/08/23 05:52:42 $

    % Grandfathered:
    % Without output arguments, IMROTATE(...) displays the rotated
    % image in the current axis.



    [so(1) so(2) thirdD] = size(A);
    [h,w,~] = size(A);
    phi = ANGLE*pi/180; % Convert to radians
    m=POINT(1,1);
    n=POINT(1,2);
    rmat = [
        cos(phi) sin(phi) 0;
        -sin(phi) cos(phi) 0;
        m-m*cos(phi)+n*sin(phi) n-m*sin(phi)-n*cos(phi) 1 
        ];
    rotate = maketform('affine',rmat);

    % Coordinates from center of A A����
    hiA = (so-1)/2;
    loA = -hiA;
    if BBOX(1)=='l' %������ Determine limits for rotated image
    hiB = ceil(max(abs(tformfwd([loA(1) hiA(2); hiA(1) hiA(2)],rotate)))/2)*2;
    loB = -hiB;
    sn = hiB - loB + 1;
    else % Cropped image����
    hiB = hiA;
    loB = loA;
    sn = so;
    end

    boxA = maketform('box',so,loA,hiA);
    boxB = maketform('box',sn,loB,hiB);
    T = maketform('composite',[fliptform(boxB),rotate,boxA]);

    if strcmp(METHOD,'bicubic')
    R = makeresampler('cubic','fill');
    elseif strcmp(METHOD,'bilinear')
    R = makeresampler('linear','fill');
    else
    R = makeresampler('nearest','fill');
    end
    B = tformarray(A, rotate, R, [1 2], [1 2], sn, [], 0);%sn������ʾ�Ĵ�С���һ��������������ɫ

%     B2 = imtransform(A, rotate, ...
%         'XData',[min(new_c(:,1)) max(new_c(:,1))],...
%         'YData',[min(new_c(:,2)) max(new_c(:,2))]);