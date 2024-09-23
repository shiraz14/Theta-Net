%% Read Image
I = imread('DIC_Source.png');
imshow(I)

%% Inject Noise to Image
J = imnoise(I,'salt & pepper',0.20);
imshow(J)

%% Save Noisy Image
imwrite(J, 'DIC_Source+Noisy.png')