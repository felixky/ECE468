%Author: Kyle Felix
%Date: 10.09.19
%Description: Filter an input image with one of the given 
%types of filters.

function spatialFiltering(filename)
[filepath,name,~] = fileparts(filename);
f = imread(filename);
f1 = im2double(f);
%imshow(f1), title('Original Image')

f2 = rgb2gray(f1);
%imshow(f2), title('Gray Scaled Image');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Laplacian Filter%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = [ -1, -1, -1; -1, 8, -1; -1, -1, -1 ];       %Laplacian 
g = f2 - imfilter(f2, w);           %Original minus edge detection
x = strcat(filepath, '/',  name);   %Fixes the issue of a missing \
x1 = strcat(x, 'Laplacian.jpg');    %concatinates the filepath, name, and adds on a fiter extension
imwrite(g, x1, 'jpg');              %Writes sharpened image to a new file

f3 = g - f2;                        %Finds the difference between the original and the sharpened
%figure, imshow(f3), title('difference');
y = strcat(x, 'LaplacianDiff.jpg');
imwrite(f3, y);              %Writes the difference to a new file

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Unsharp Masking Filter%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = fspecial('gaussian', [5 5],10 );       %Gaussian filter 
g = f2 + (f2 - imfilter(f2, w));    %Original plus (Original minus edge detection)
x = strcat(filepath, '/',  name);   %Fixes the issue of a missing \
x1 = strcat(x, 'Unsharp.jpg');    %concatinates the filepath, name, and adds on a fiter extension
imwrite(g, x1, 'jpg');              %Writes sharpened image to a new file

f3 = g - f2;                        %Finds the difference between the original and the sharpened
%figure, imshow(f3), title('difference');
y = strcat(x, 'UnsharpDiff.jpg');
imwrite(f3, y);              %Writes the difference to a new file

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Image averaging + Laplacian Filter%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = (1/9)*[1, 1, 1; 1, 1, 1; 1, 1, 1];  %Averaging Filter
T = imfilter(f2, k); %Convolution of grayscale image and averaging filter 
w = fspecial('laplacian', 0);
g = f2 - imfilter(T, w);           %Original minus edge detection
x = strcat(filepath, '/', name);   %Fixes the issue of a missing \
x1 = strcat(x, 'AvgLaplace.jpg');    %concatinates the filepath, name, and adds on a fiter extension
imwrite(g, x1, 'jpg');              %Writes sharpened image to a new file

f3 = g - f2;                        %Finds the difference between the original and the sharpened
%figure, imshow(f3), title('difference');
y = strcat(x, 'AvgLaplaceDiff.jpg');
imwrite(f3, y);              %Writes the difference to a new file
return