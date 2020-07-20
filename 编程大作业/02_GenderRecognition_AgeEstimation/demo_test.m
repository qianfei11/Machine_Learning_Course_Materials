%%%%%% This is a matlab interface used to test %%%%%%%
%%%%%% You need to fill out thiss code %%%%%%
%%%%%% You need to make this code runable!!! %%%%%
clear;clc;

%%%%%%%%%%% Section 1.Steps of Running environment configuration %%%%%%%%%%
%%%%%%%%%%%% If you need, fill it;otherwise, skip it %%%%%%%%%%

%%%%%%%%%%%%End of section 1%%%%%%%%%%%

input_dir = '';%%%% absolute path of test images
fid1 = fopen('result.txt', 'w');
subdir = dir(input_dir);
for i=3: length(subdir)
    if ~ subdir(i).isdir
        continue;
    end
    img_fns = dir([input_dir '/' subdir(i).name '/*.jpg']);
    for j = 1:length(img_fns)
        img = imread([input_dir '/' subdir(i).name '/' img_fns(j).name]);
        
        %%%%%%%%%%% Section 2.Steps of Image Preprocessing %%%%%%%%%%
        %%%%%%%%%%%% If you need, fill it;otherwise, skip it %%%%%%%%%%
        
        %%%%%%%%%%%%End of section 2%%%%%%%%%%%
        
        %%%%%%%%%%% Section 3.Steps of Prediction %%%%%%%%%%
        %%%%%%%%%%%% Generally you do not need to revise %%%%%%%%%%
        age = AgeEstimation(img);
        gender = GenderIdentification(img);
        %%%%%%%%%%%%End of section 2%%%%%%%%%%%
        
        fprintf(fid1, '%s %d %d\n', img_fns(j).name, age, gender);
    end
end
fclose(fid1);

function age = AgeEstimation(image);
%%%%%%%%%%%%%You need to fill out this section %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% This section contains predict steps%%%%%%%%%%%%%%%%%%

end

function gender = GenderIdentification(image);
%%%%%%%%%%%%%You need to fill out this section %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% This section contains predict steps%%%%%%%%%%%%%%%%%%

end