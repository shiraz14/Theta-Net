% Read the images
drefa = imread("PCM (3) - Target (5A7).png");
drefb = imread("PCM (3) - Target (6A8).png");

OGENA = imread("PCM (3) - O-Net (5).png");
TGENA = imread("PCM (3) - Theta-Net (7).png");

OGENB = imread("PCM (3) - O-Net (6).png");
TGENB = imread("PCM (3) - Theta-Net (8).png");

%% Find the SSIM (global) & SSIM (local) for O-Net generated DIC image ROI1
[dssimval,dssimmap] = ssim(OGENA,drefa);
imshow(dssimmap,[])
title("Local O-Net SSIM Map with Global SSIM Value: "+num2str(dssimval))

%% Find the SSIM (global) & SSIM (local) for Theta-Net generated DIC image ROI1
[dssimvala,dssimmapa] = ssim(TGENA,drefa);
imshow(dssimmapa,[])
title("Local Theta-Net SSIM Map with Global SSIM Value: "+num2str(dssimvala))

%% Find the SSIM (global) & SSIM (local) for O-Net generated DIC image ROI2
[pssimval,pssimmap] = ssim(OGENB,drefb);
imshow(pssimmap,[])
title("Local O-Net SSIM Map with Global SSIM Value: "+num2str(pssimval))

%% Find the SSIM (global) & SSIM (local) for Theta-Net generated DIC image ROI2
[pssimvala,pssimmapa] = ssim(TGENB,drefb);
imshow(pssimmapa,[])
title("Local Theta-Net SSIM Map with Global SSIM Value: "+num2str(pssimvala))

%% Calculate PSNR/SNR
[peaksnr, snr] = psnr(OGENA, drefa);
[peaksnr2, snr2] = psnr(TGENA, drefa);
[peaksnr3, snr3] = psnr(OGENB, drefb);
[peaksnr4, snr4] = psnr(TGENB, drefb);

%% Calculate IMSE (https://www.mathworks.com/help/images/ref/immse.html)
err = immse(OGENA, drefa);
err2 = immse(TGENA, drefa);
err3 = immse(OGENB, drefb);
err4 = immse(TGENB, drefb);