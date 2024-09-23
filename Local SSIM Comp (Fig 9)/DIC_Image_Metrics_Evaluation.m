% Read the images
drefa = imread("DIC (2) - Target (1A3).png");
drefb = imread("DIC (2) - Target (2A4).png");

OGENA = imread("DIC (2) - O-Net (1).png");
TGENA = imread("DIC (2) - Theta-Net (3).png");

OGENB = imread("DIC (2) - O-Net (2).png");
TGENB = imread("DIC (2) - Theta-Net (4).png");

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