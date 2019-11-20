%Author: Kyle Felix
%Date: 11/20/19
%Description:  Wiener filter

function wienerImage(filename, xspd, yspd, sigma)
clc
[filepath,name,ext] = fileparts(filename);
i1 = im2double(imread(filename));   %read in filename and turn the pixel values into doubles

if 3 == size(i1,3)      %If it is a color image
   f = rgb2gray(i1);                   %grayscale the image
else                    %if already a greyscale
    f = i1;
end

T = .01;
a = xspd;
b = yspd;
F = fft2(f);                        %2D Fourier Trandform of corrupted image
sz = size(F);
u = sz(1);
v = sz(2);

W = zeros(u,v);
H = zeros(u,v);

SNR = (sigma^2)/(255^2);
for u = 1:sz(1)
    for v = 1:sz(2)
       H(u,v) = (T/((pi)*(u*a+v*b)))*(sin((pi)*(u*a+u*b))*exp((-1i)*(pi)*(u*a+v*b))); %inverse filter. Part of Weiner filter
       W(u,v) = ((conj(H(u,v)))./((H(u,v).^2)+(1/SNR)));        %Wiener filter
    end
end   

G = F.*W;                           %Corrupted image times wiener filter in frequency domain
g = ifft2(G);%inverse fft of the restored image
t = imadjust(real(g));
gg = t + (g - real(g));
new_name = strcat(name, 'Wiener.jpg');     %appending Wiener to the end of the filename
imwrite(real(gg), new_name);                     %Write new image to file in directory

return