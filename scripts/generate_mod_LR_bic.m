function generate_mod_LR_bic()

%% set parameters
% comment the unnecessary line
test_image = {'Set5', 'Set14', 'B100', 'Urban100'};

up_scale = 4;
for ii = 1 : size(test_image, 2)
input_path = ['/home/ser606/Documents/neuro/datasets/TestDataSR/',test_image{ii}];

save_bic_path = strcat(input_path, '_LRBicx', num2str(up_scale));

if exist('save_mod_path', 'var')
    if exist(save_mod_path, 'dir')
        disp(['It will cover ', save_mod_path]);
    else
        mkdir(save_mod_path);
    end
end
if exist('save_LR_path', 'var')
    if exist(save_LR_path, 'dir')
        disp(['It will cover ', save_LR_path]);
    else
        mkdir(save_LR_path);
    end
end
if exist('save_bic_path', 'var')
    if exist(save_bic_path, 'dir')
        disp(['It will cover ', save_bic_path]);
    else
        mkdir(save_bic_path);
    end
end
if exist('save_LRBD_path', 'var')
    if exist(save_LRBD_path, 'dir')
        disp(['It will cover ', save_LRBD_path]);
    else
        mkdir(save_LRBD_path);
    end
end
if exist('save_LRDN_path', 'var')
    if exist(save_LRDN_path, 'dir')
        disp(['It will cover ', save_LRDN_path]);
    else
        mkdir(save_LRDN_path);
    end
end
if exist('save_LRBD_bic_path', 'var')
    if exist(save_LRBD_bic_path, 'dir')
        disp(['It will cover ', save_LRBD_bic_path]);
    else
        mkdir(save_LRBD_bic_path);
    end
end


idx = 0;
filepaths = dir(fullfile(input_path,'*.*'));
for i = 1 : length(filepaths)
    [paths,imname,ext] = fileparts(filepaths(i).name);
    if isempty(imname)
        disp('Ignore . folder.');
    elseif strcmp(imname, '.')
        disp('Ignore .. folder.');
    else
        idx = idx + 1;
        str_rlt = sprintf('%d\t%s.\n', idx, imname);
        fprintf(str_rlt);
        % read image
        img = imread(fullfile(input_path, [imname, ext]));
        % modcrop
        img = modcrop(img, up_scale);
        if exist('save_mod_path', 'var')
            imwrite(img, fullfile(save_mod_path, [imname, '.png']));
        end
        % LR
        if exist('save_LRBD_path', 'var')
            kernel = fspecial('gaussian', 7, 1.6); 
            im_LR = imfilter(img, kernel, 'replicate');
            im_LR = imresize(im_LR, 1/up_scale, 'nearest');
            im_LR = imresize(im_LR, up_scale, 'bicubic');
            imwrite(im_LR, fullfile(save_LRBD_path, [imname, '.png']));
        end
        
        if exist('save_LRDN_path', 'var')
            im_LR = single(imresize(img, 1/up_scale, 'bicubic'));
            im_LR = uint8(im_LR + single(30*randn(size(im_LR))));
            im_LR = imresize(im_LR, up_scale, 'bicubic');
            imwrite(im_LR, fullfile(save_LRDN_path, [imname, '.png']));
        end       
        
        
        im_LR = imresize(img, 1/up_scale, 'bicubic');
        if exist('save_LR_path', 'var')
            imwrite(im_LR, fullfile(save_LR_path, [imname, '_LRx', num2str(up_scale), '.png']));
        end
        if exist('save_bic_path', 'var')
            im_B = imresize(im_LR, up_scale, 'bicubic');
            imwrite(im_B, fullfile(save_bic_path, [imname, '_bicx', num2str(up_scale), '.png']));
        end
    end
end
end
end

%% modcrop
function imgs = modcrop(imgs, modulo)
if size(imgs,3)==1
    sz = size(imgs);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2));
else
    tmpsz = size(imgs);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2),:);
end
end
