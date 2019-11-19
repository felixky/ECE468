%Author: Kyle Felix
%Date: 10/23/19
%Description:  

function wienerImage(filename, xspd, yspd, sigma)
[filepath,name,ext] = fileparts(filename);
i1 = im2double(imread(filename));   %read in filename and turn the pixel values into doubles
f = rgb2gray(i1);                   %grayscale the image
F = fft2(f);                        %2D Fourier Trandform of corrupted image

H =                                 %inverse filter. Part of Weiner filter
W =                                 %Wiener filter
G = F.*W;                           %Corrupted image times wiener filter in frequency domain

g = ifft2(G);                       %inverse fft of the restored image

x = strcat(name, 'Wiener.jpg');     %appending Wiener to the end of the filename
imwrite(g, x);                     %Write new image to file in directory

return