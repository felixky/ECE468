%Author: Kyle Felix
%Date: 10/23/19
%Description:  2D Fourier Transforms Manipulation

function twoDFourier(filename)
[filepath,name,ext] = fileparts(filename);
i1 = im2double(imread(filename));   %read in filename and turn the pixel values into doubles
i2 = rgb2gray(i1);                  %grayscale the image
                                    %imshow(i2), title('Original Image');
PQ = 2*size(i2);                    %Find the dimensions of the original image time 2 for padding
XY  = size(i2);                     %Finds the size of the original image

DFT = fft2(i2,PQ(1), PQ(2));        %Discrete Fourier Transform of the grayscaled image 
% that is padded so that its dimensions are twice that of the original image
 
%5.1 - Set the magnitude of Fourier Transform to 1 and save the inverse in a file
%"filename1.jpg"

DFT2 = fftshift(DFT);               %Moves the DC frequency values to the center of the image
M = angle(DFT);                     %M is the phase component of the DFT
f1 = ifft2(M);
F = (255.*(f1 - min(min(f1)))./(max(max(f1))));  %Normalized to display the magnitude
imshow(abs(DFT2)), title('Magnitude of DFT');

F2 = imcrop(real(F), [0 0 XY(2) XY(1)]);    %Crops the image to the original size
x = strcat(name, '1.jpg');          %appending 1 to the end of the filename
imwrite(real(F2), x);               %Write new image to file in directory

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%5.2 - set the phase of the Fourier Transform equal to zero and save the
%inverse of image in "filename2.jpg"
T = abs(DFT);                       %T is the magnitude of the image without a phase component
f2 = ifft2(T);                      %Inverse DFT
F2 = imcrop(real(f2), [0 0 XY(2) XY(1)]);   %Crops the image to the original size
x = strcat(name, '2.jpg');          %appending 1 to the end of the filename
imwrite(F2, x);               %Write new image to file in directory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%5.3 - save the inverse of shifted fourier transform in "filename3.jpg"
DR = size(DFT);                     %Finds the size of the DFT aka u and v
Y = DFT.*((-1)^(DR(1)+DR(2)));
f3 = ifft2(Y);                      %Inverse DFT


F2 = imcrop(real(f3), [0 0 XY(2) XY(1)]);   %Crops the image to the original size

x = strcat(name, '3.jpg');          %appending 1 to the end of the filename
imwrite(real(F2), x);               %Write new image to file in directory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%5.4 - Save the inverse of flipped F(-u,-v) into "filename4.jpg" 
a = circshift(fliplr(DFT), [0 1]);  %circshift 1 column to the right to re-align image
b = circshift(flipud(a), [1 0]);    %circshift 1 row up to re-align image
f5 = circshift(ifft2(b), [0 -1]);   %circshift 1 column to the left to re-align image
F2 = imcrop(real(f5), [XY(2) XY(1) PQ(1) PQ(2)]);%Crops the image to the original size
x = strcat(name, '4.jpg');          %appending 1 to the end of the filename
imwrite(real(F2), x);               %Write new image to file in directory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


return