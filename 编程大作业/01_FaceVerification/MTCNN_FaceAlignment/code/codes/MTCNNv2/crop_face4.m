function [ face ] = crop_face4( img, bbox, points,face_size)
%CROP_FACE Summary of this function goes here
%   Detailed explanation goes here
   %image center

   [mh,mw,~] = size(img);
   x1 = points(1);
   x2 = points(2);
   y1 = points(6);
   y2 = points(7);
   %eye center
   x0 = (x1 + x2)/2;
   y0 = (y1+y2)/2;

   theta = atan((y2-y1)/(x2-x1));
   theta = double(theta);
%    im_rotate = imrotatep(double(img),theta*180/pi,'bilinear','loose',[double(x0),double(y0)]);

    %% rotate the image
   rmat = [
       cos(-theta) sin(-theta) 0;
       -sin(-theta) cos(-theta) 0;
       0 0 1
       ];
   corners = [
       0 0 1
       mw 0 1
       0 mh 1
       mw mh 1
       ];
   new_c = corners*rmat;
   T = maketform('affine',rmat);
   newCenter = [x0,y0,1]*rmat - min(new_c);
   im_rotate2 = imtransform(img,T,...
       'XData',[min(new_c(:,1)) max(new_c(:,1))],...
       'YData',[min(new_c(:,2)) max(new_c(:,2))]);
   
   [r_h,r_w,c] = size(im_rotate2);
   face_h = bbox(4)-bbox(2);
   face_w = bbox(3)-bbox(1);
%  Make sure height/width = 55/47
   if (bbox(4)-bbox(2))/(bbox(3)-bbox(1))>55/47
       face_w = (bbox(4)-bbox(2))*47/55;
   else
       face_h = (bbox(3)-bbox(1))*55/47;
   end
   face_w = face_w*face_size;
   face_h = face_h*face_size;
   %Calculate face boundingbox
   bbox(1) = newCenter(1) - face_w/2;
   bbox(3) = newCenter(1) + face_w/2;
   bbox(2) = newCenter(2) - face_h*0.4;
   bbox(4) = newCenter(2) + face_h*0.6;
   %Clip the face boundingbox
   bbox(1) = max(bbox(1),1);
   bbox(2) = max(bbox(2),1);
   bbox(3) = min(bbox(3),r_w);
   bbox(4) = min(bbox(4),r_h);
   
   bbox = round(bbox(:,1:4));
   face = uint8(im_rotate2(bbox(2):bbox(4),bbox(1):bbox(3),:));
%    imshow(face);
%    figure(2);
%    imshow(uint8(im_rotate2));
%    hold on;
%    plot(newCenter(1),newCenter(2),'g.','MarkerSize',10);
end

