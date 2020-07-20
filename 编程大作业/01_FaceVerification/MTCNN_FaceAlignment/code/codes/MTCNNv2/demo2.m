clear;clc;

%minimum size of face
minsize=20;

%path of toolbox
caffe_path='~/Caffe/Caffe_default/matlab';
pdollar_toolbox_path='../toolbox-master'
caffe_model_path='./model'
addpath(genpath(caffe_path));
addpath(genpath(pdollar_toolbox_path));

%use cpu
%caffe.set_mode_cpu();
gpu_id=2;
caffe.set_mode_gpu();	
caffe.set_device(gpu_id);

%three steps's threshold
threshold=[0.6 0.7 0.7]

%scale factor
factor=0.709;


%load caffe models
prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
model_dir = strcat(caffe_model_path,'/det1.caffemodel');
PNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
model_dir = strcat(caffe_model_path,'/det2.caffemodel');
RNet=caffe.Net(prototxt_dir,model_dir,'test');	
prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
model_dir = strcat(caffe_model_path,'/det3.caffemodel');
ONet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir =  strcat(caffe_model_path,'/det4.prototxt');
model_dir =  strcat(caffe_model_path,'/det4.caffemodel');
LNet=caffe.Net(prototxt_dir,model_dir,'test');
faces=cell(0);	
input_dir = '~/Caffe/MTCNN_FaceAlignment/webface_samples'
output_dir = '~/Caffe/MTCNN_FaceAlignment/webface_samples_aligned'

if ~exist(output_dir)
    mkdir(output_dir)
end
fid = fopen([output_dir '/undetected.txt'],'w'); 
folders = dir(input_dir);
for i=3:size(folders,1)
    i
    folder_dir = [input_dir '/' folders(i).name];
    output_folder_dir = [output_dir '/' folders(i).name];
    if ~exist(output_folder_dir)
        mkdir(output_folder_dir);
    end

    names = dir(folder_dir);
    for j=3:size(names,1)
        output_img_dir = [output_folder_dir '/' names(j).name];
        img_dir = [folder_dir '/' names(j).name];
        try
            img = imread(img_dir);
        catch
            fprintf(fid,'%s\n',img_dir);
            continue;
        end

        if size(img,3)==1
            fprintf(fid,'%s\n',img_dir);
            continue;
        end
        [boudingboxes points]=detect_face(img,minsize,PNet,RNet,ONet,LNet,threshold,false,factor);
        %show detection result
        numbox=size(boudingboxes,1);
        if ~numbox
            [img_dir ' undetected.']
            fprintf(fid,'%s\n',img_dir);
            continue;
        end
        figure(1);
        %imshow(img)

        hold on;
        for k = 1:numbox
            plot(points(1:5,k),points(6:10,k),'g.','MarkerSize',10);
            r=rectangle('Position',[boudingboxes(k,1:2) boudingboxes(k,3:4)-boudingboxes(k,1:2)],'Edgecolor','g','LineWidth',3);
        end
        max_idx = 1;
        max_size = 1;
        [height,width,~] = size(img);
        for k = 1:numbox
            s = (min(boudingboxes(k,3),width) - max(boudingboxes(k,1),1))*...
                (min(boudingboxes(k,4),height) -max(boudingboxes(k,2),1));
            if max_size < s
                max_idx = k;
                max_size = s;
            end
        end
        im_face = crop_face4(img, boudingboxes(max_idx,:), points(1:10,max_idx),1);

        if size(im_face,2) < 10
            [img_dir ' undetected.']
            fprintf(fid,'%s\n',img_dir);
            continue;
        end
        figure(3);
        %imshow(im_face);
        imwrite(im_face, output_img_dir);
        % hold off; 
        % pause
    end
    % fclose(fid1);
end

fclose(fid);
