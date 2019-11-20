%Author: Kyle Felix
%Date: 10/23/19
%Description: 

function DenoiseImage(filename)
clc
Eg = [0, 0, 0, 0];
Es = [0, 0, 0, 0];

[filepath,name,ext] = fileparts(filename);
f = im2double(imread(filename));   %read in filename and turn the pixel values into doubles
f = rgb2gray(f);                   %grayscale the image
f1 = f.*255;                        %Scaled image
imwrite(f, 'imagegrey.jpg');        %Write new image to file in directory
sz = size(f);%Finding the size of the input image
x = sz(1);
y = sz(2);
fs = zeros(sz(1),sz(2));            %matrix of zeros for later

x2 = randn(sz(1),sz(2));       %Random matrix the same size as the image
eta = 100.*(x2./max(max(x2)));       %Normalize the range from 0-1. Mean=0. set variance to 100

fgg = f1 + eta;                       %adding the gaussian noise to the original image
fg = fgg./255;                      %Normalize the image back to a range of 0-1

new_name = strcat(name, 'GaussianNoise.jpg');          %appending 1 to the end of the filename
imwrite(fg, new_name);                     %Write new image to file in directory

t1 = 255.*abs(rand(sz(1),sz(2)));
t2 = 255.*abs(rand(sz(1),sz(2)));

for c = 1:sz(1)         %for each row
    for s = 1:sz(2)       %for each column
        if f1(c,s) > t1(c,s) %compare image to t1 at every pixel
            fs(c,s) = 255;
        elseif f1(c,s) < t2(c,s)
            fs(c,s) = 0;
        else% f1(c,s) < t1(c,s) %& (f1(c,s) > t2(c,s)))
            fs(c,s) = f1(c,s);         
        end
    end
end
fs = fs./255;
fss = fs.*255;
new_name = strcat(name, 'SaltNoise.jpg');          %appending 1 to the end of the filename
imwrite(fs, new_name);                     %Write new image to file in directory

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Defining the 4 filters and writing the filtered images to new files
%Filters for fg, gaussian noise
wg1 = fspecial('gaussian',[3 3], 1);     %Gaussian filter with sigma=1
gg1 = imfilter(fg,wg1);
gg2 = medfilt2(fg, [3 3]);  %Median Filter
gg3 = wiener2(fg,[3 3]);    %Wiener Filter
%Adaptive Filter
%d = [0,0,0;0,1,0;0,0,0];
d = [255];
fp = zeros(sz(1)+2,sz(2)+2);
for c = 1:sz(1)         %for each row
    for s = 1:sz(2)       %for each column
        fp(c+1,s+1) = fgg(c,s);      
    end
end
wg4 = zeros(sz(1),sz(2));
w = zeros(sz(1),sz(2));
for c = 2:sz(1)+1         %for each row
    for s = 2:sz(2)+1       %for each column
        am = 0;
        av =0;
        for i = -1:1
           for j = -1:1
              am = am + fp(c+i,s+j); 
              av = av + fp(c+i,s+j); 
           end
       end    
       am = (1/9).*am;
       av = (1/9).*((av-am)^2);
       wg4(c-1,s-1) = fgg(c-1,s-1) - ((5000/av)*(fgg(c-1,s-1)-am));
       w(c-1,s-1) = ((5000/av)*(fgg(c-1,s-1)-am));
    end
end 
imhist(w);
gg4 = wg4./255;

new_name = strcat(name, 'GaussianGaussian.jpg');          %appending 1 to the end of the filename
imwrite(gg1, new_name);                     %Write new image to file in directory
new_name = strcat(name, 'GaussianMedian.jpg');          %appending 1 to the end of the filename
imwrite(gg2, new_name);                     %Write new image to file in directory
new_name = strcat(name, 'GaussianWiener.jpg');          %appending 1 to the end of the filename
imwrite(gg3, new_name);                     %Write new image to file in directory
new_name = strcat(name, 'GaussianAdaptive.jpg');          %appending 1 to the end of the filename
imwrite(gg4, new_name);                     %Write new image to file in directory


%Filters for fs, salt & pepper noise
ws1 = fspecial('gaussian',[3 3], 1);     %Gaussian filter with sigma=1
gs1 = imfilter(fs,ws1);
gs2 = medfilt2(fs, [3 3]);
gs3 = wiener2(fs,[3 3]);

fp = zeros(sz(1)+2,sz(2)+2);
for c = 1:sz(1)         %for each row
    for s = 1:sz(2)       %for each column
        fp(c+1,s+1) = f1(c,s);      
    end
end

ws4 = zeros(sz(1),sz(2));
for c = 2:sz(1)+1         %for each row
    for s = 2:sz(2)+1       %for each column
        am = 0;
        av =0;
        for i = -1:1
           for j = -1:1
              am = am + fp(c+i,s+j); 
              av = av + fp(c+i,s+j); 
           end
       end    
       am = (1/9).*am;
       av = (1/9).*((av-am)^2);
       ws4(c-1,s-1) = fss(c-1,s-1) + ((5000/av)*(fss(c-1,s-1)-am));
    end
end 
gs4 = ws4./255;

new_name = strcat(name, 'SaltGaussian.jpg');          %appending 1 to the end of the filename
imwrite(gs1, new_name);                     %Write new image to file in directory
new_name = strcat(name, 'SaltMedian.jpg');          %appending 1 to the end of the filename
imwrite(gs2, new_name);                     %Write new image to file in directory
new_name = strcat(name, 'SaltWiener.jpg');          %appending 1 to the end of the filename
imwrite(gs3, new_name);                     %Write new image to file in directory
new_name = strcat(name, 'SaltAdaptive.jpg');          %appending 1 to the end of the filename
imwrite(gs4, new_name);                     %Write new image to file in directory

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculating the error for each filter type
%Gaussian Noise
for c = 1:sz(1)         %for each row
    for s = 1:sz(2)       %for each column
        Eg(1) = Eg(1) + (f(c,s) - gg1(c,s))^2;       
    end
end
for c = 1:sz(1)         %for each row
    for s = 1:sz(2)       %for each column
        Eg(2) = Eg(2) + (f(c,s) - gg2(c,s))^2;       
    end
end
for c = 1:sz(1)         %for each row
    for s = 1:sz(2)       %for each column
        Eg(3) = Eg(3) + (f(c,s) - gg3(c,s))^2;       
    end
end
for c = 1:sz(1)         %for each row
    for s = 1:sz(2)       %for each column
        Eg(4) = Eg(4) + (f(c,s) - gg4(c,s))^2;       
    end
end

%Salt & Pepper Noise
for c = 1:sz(1)         %for each row
    for s = 1:sz(2)       %for each column
        Es(1) = Es(1) + (f(c,s) - gs1(c,s))^2;       
    end
end
for c = 1:sz(1)         %for each row
    for s = 1:sz(2)       %for each column
        Es(2) = Es(2) + (f(c,s) - gs2(c,s))^2;       
    end
end
for c = 1:sz(1)         %for each row
    for s = 1:sz(2)       %for each column
        Es(3) = Es(3) + (f(c,s) - gs3(c,s))^2;       
    end
end
for c = 1:sz(1)         %for each row
    for s = 1:sz(2)       %for each column
        Es(4) = Es(4) + (f(c,s) - gs4(c,s))^2;       
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Displaying the best filter for each type of noise
disp('Best Filter for Gaussian Noise: ')
[M,I] = min(Eg);
if I == 1
    disp('Gaussian')
elseif I == 2
    disp('Median')
elseif I == 3
    disp('Weiner')
else
    disp('Adaptive')
end
% Eg
%min(Eg)
disp('')
disp('Best Filter for Salt & Pepper Noise: ')
[M,I] = min(Es);
if I == 1
    disp('Gaussian')
elseif I == 2
    disp('Median')
elseif I == 3
    disp('Weiner')
else
    disp('Adaptive')
end
% Es
% min(Es)
return