clear 
close all
%% all read image and feature
index_img = dir('C:\Users\104viplab\Desktop\Skin_Color_code\train_set\*.jpg');
index_ppm = dir('C:\Users\104viplab\Desktop\Skin_Color_code\train_set\*.ppm');
merge_skin_ycbcr = zeros(1,3);
count = 0;
for i=1:size(index_img,1)
    if i ~= 0
        temp_image = imread(['C:\Users\104viplab\Desktop\Skin_Color_code\Skin-Color-Model-master\lfw_funneled\Aaron_Peirsol\' index_img(i).name]);
        temp_ppm   = imread(['C:\Users\104viplab\Desktop\Skin_Color_code\Skin-Color-Model-master\lfw_funneled\Aaron_Peirsol\' index_ppm(i).name]);
        temp_image = rgb2ycbcr(temp_image);
        temp_imageL=reshape(temp_image, size(temp_image,1)*size(temp_image,2),3);
        feature_ppm = temp_ppm(:,:,2);
        [loc]= find(feature_ppm);
        skin_ycbcr = double(temp_imageL(loc, :));
        
        for j = 1 : size(skin_ycbcr,1)
            if j ~=0
            count = count+1;
            merge_skin_ycbcr(count,:,:) = skin_ycbcr(j,:,:);
            end
        end
    end
end
merge_skin_ycbcr =double(merge_skin_ycbcr);
%% show image
figure; 
scatter3(merge_skin_ycbcr(:,1),merge_skin_ycbcr(:,2), merge_skin_ycbcr(:,3),'r.');
xlabel('Y'), ylabel('Cb'), zlabel('Cr')

%% max and min
Y = merge_skin_ycbcr(:,1);
maxY=max(Y);
minY=min(Y);
DeltaY = round(maxY-minY)/3;
%% assign cluster
cluster_index=find(Y < minY + DeltaY);
E = zeros(59344,1);
E(cluster_index) = 1;
cluster_index=find((Y >= minY + DeltaY) & (Y < minY + DeltaY*2));
E(cluster_index) = 2;
cluster_index=find(Y >= minY+DeltaY*2);
E(cluster_index) = 3;
%% train
% Get centroid and covariance matrix
for i=min(E):max(E)
    cluster_index=find(E == i);
    cluster_color = merge_skin_ycbcr(cluster_index,:);
    M(i,:) = mean(cluster_color);
    delta = cluster_color - repmat(M(i,:), size(cluster_color,1), 1);
    C(i,:,:) = (delta' * delta)/(size(delta,1)-1);
end
distance = zeros(59344,3);
New_M = zeros(3,3);
% repeated train
for iter=1:1000
    if iter ~= 0
        disp(['iter = ' num2str(iter)])
% convergence mahalanobis         
        for j=1:size(M,1)
           m=M(j,:);
           c=reshape(C(j,:,:), 3, 3);
           delta = merge_skin_ycbcr - repmat(m, size(merge_skin_ycbcr,1), 1);
           distance(:,j) = sum(delta*inv(c) .* delta, 2);
        end
        [D E] = min(distance,[],2);
        for t=min(E):max(E)
            cluster_index=find(E == t);
            cluster_color = merge_skin_ycbcr(cluster_index,:);
            M(t,:) = mean(cluster_color);
            delta = cluster_color - repmat(M(t,:), size(cluster_color,1), 1);
            C(t,:,:) = (delta' * delta)/(size(delta,1)-1);
        end
        if iter == 1 || iter == 10 || iter == 25 || iter == 50 || min(min(New_M)) == min(min(M))
            figure, axis([16 250 16 200 16 250]), xlabel('Y'), ylabel('Cb'), zlabel('Cr'),title(['step-2 train = ' num2str(iter)]), hold on;
            for j=1:size(M,1)
                c = reshape(C(j,:,:), 3, 3);
                plot_gaussian_ellipsoid(M(j,:), c);
            end
        end
% convergence
        if New_M == M
            disp(['convergence iter = ' num2str(iter)])
            break 
        end
        New_M = M ;
    end
end
%% test image skin color detection
% all read image and feature
test_index_img = dir('C:\Users\104viplab\Desktop\Skin_Color_code\test_set\*.jpg');
test_index_ppm = dir('C:\Users\104viplab\Desktop\Skin_Color_code\test_set\*.ppm');
count = 0;
for test_i=1:size(test_index_img,1)
    if test_i ~= 0
        temp_test_image = imread(['C:\Users\104viplab\Desktop\Skin_Color_code\test_set\' test_index_img(test_i).name]);
        temp_test_ppm   = imread(['C:\Users\104viplab\Desktop\Skin_Color_code\test_set\' test_index_ppm(test_i).name]);
        
        test_img_YCbCr  = rgb2ycbcr(temp_test_image);
        temp_test_imageL=reshape(test_img_YCbCr, size(test_img_YCbCr,1)*size(test_img_YCbCr,2),3);
        temp_test_imageL=double(temp_test_imageL);
        
        feature_test_ppm = temp_test_ppm(:,:,2);
        [test_loc]= find(feature_test_ppm);
        skin_ycbcr = double(temp_test_imageL(test_loc, :));
% convergence mahalanobis        
        for test_j=1:size(M,1)
            test_m = M(test_j,:);
            test_c = reshape(C(test_j,:,:), 3, 3); 
            test_delta = double(temp_test_imageL) - repmat(test_m, size(temp_test_imageL,1), 1);
            test_distance(:,test_j) = sum(test_delta*inv(test_c) .* test_delta, 2);
        end
        dist_min = min(test_distance, [], 2);
% calculate skin_map
        skin_pt=find(dist_min < 4);
        skin_map = zeros(size(temp_test_image,1)*size(temp_test_image,2),1);
        skin_map(skin_pt,1) = 1;
        skin_map = reshape(skin_map, size(temp_test_image,1), size(temp_test_image,2));
        figure,
        subplot(1,3,1),imshow(skin_map,[]),title(['test ' num2str(test_i) ' skin color map']);
% calculate recall and precision
        total_and=bitand(skin_map,(feature_test_ppm~=0));
        recall=size(find(total_and))/size(find(feature_test_ppm~=0));
        precision = size(find(total_and))/size(find(skin_map));
        subplot(1,3,2),imshow(uint8(temp_test_image)),title('original image');
% map and original        
        temp_test_image = double(temp_test_image);
        step_two_img(:,:,1)=(skin_map.*temp_test_image(:,:,1));%step 2 and
        step_two_img(:,:,2)=(skin_map.*temp_test_image(:,:,2));
        step_two_img(:,:,3)=(skin_map.*temp_test_image(:,:,3));
        subplot(1,3,3),imshow(uint8(step_two_img),[]),title(' map and original result '  );
        disp(['img ' num2str(test_i) ' Recall ' num2str(recall) ' Precision = ' num2str(precision) ])
    end
end
