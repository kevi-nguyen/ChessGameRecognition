import cv2
import numpy as np


class Retinex:
    def get_ksize(self, sigma):
        return int(((sigma - 0.8) / 0.15) + 2.0)

    def get_gaussian_blur(self, img, ksize=0, sigma=5):
        if ksize == 0:
            ksize = self.get_ksize(sigma)

        # Gaussian 2D-kernel can be seperable into 2-orthogonal vectors
        # then compute full kernel by taking outer product or simply mul(V, V.T)
        sep_k = cv2.getGaussianKernel(ksize, sigma)

        # if ksize >= 11, then convolution is computed by applying fourier transform
        return cv2.filter2D(img, -1, np.outer(sep_k, sep_k))

    def ssr(self, img, sigma):
        # Single-scale retinex of an image
        # SSR(x, y) = log(I(x, y)) - log(I(x, y)*F(x, y))
        # F = surrounding function, here Gaussian

        return np.log10(img) - np.log10(self.get_gaussian_blur(img, ksize=0, sigma=sigma) + 1.0)

    def msr(self, img, sigma_scales=[15, 80, 250], apply_normalization=True):
        img = img.astype(np.float64) + 1.0

        # Multi-scale retinex of an image
        # MSR(x,y) = sum(weight[i]*SSR(x,y, scale[i])), i = {1..n} scales

        msr = np.zeros(img.shape)

        for sigma in sigma_scales:
            msr += self.ssr(img, sigma)

        msr = msr / len(sigma_scales)

        if apply_normalization:
            msr = cv2.normalize(msr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

        return msr

    def color_balance(self, img, low_per, high_per):
        '''Contrast stretch img by histogram equilization with black and white cap'''

        tot_pix = img.shape[1] * img.shape[0]
        low_count = tot_pix * low_per / 100
        high_count = tot_pix * (100 - high_per) / 100

        # channels of image
        ch_list = []
        if len(img.shape) == 2:
            ch_list = [img]
        else:
            ch_list = cv2.split(img)

        cs_img = []
        for i in range(len(ch_list)):
            ch = ch_list[i]
            cum_hist_sum = np.cumsum(cv2.calcHist([ch], [0], None, [256], (0, 256)))

            li, hi = np.searchsorted(cum_hist_sum, (low_count, high_count))
            if (li == hi):
                cs_img.append(ch)
                continue

            lut = np.array([0 if i < li
                            else (255 if i > hi else round((i - li) / (hi - li) * 255))
                            for i in np.arange(0, 256)], dtype='uint8')
            cs_ch = cv2.LUT(ch, lut)
            cs_img.append(cs_ch)

        if len(cs_img) == 1:
            return np.squeeze(cs_img)
        elif len(cs_img) > 1:
            return cv2.merge(cs_img)
        return None

    def msrcr(self, img, sigma_scales=[15, 80, 250], alpha=125, beta=46, G=192, b=-30, low_per=1, high_per=1):
        # Multi-scale retinex with Color Restoration
        # MSRCR(x,y) = G * [MSR(x,y)*CRF(x,y) - b], G=gain and b=offset
        # CRF(x,y) = beta*[log(alpha*I(x,y) - log(I'(x,y))]
        # I'(x,y) = sum(Ic(x,y)), c={0...k-1}, k=no.of channels

        img = img.astype(np.float64) + 1.0

        msr_img = self.msr(img, sigma_scales, apply_normalization=False)
        # Color-restoration function
        crf = beta * (np.log10(alpha * img) - np.log10(np.sum(img, axis=2, keepdims=True)))
        # MSRCR
        msrcr = G * (msr_img * crf - b)
        # normalize MSRCR
        msrcr = cv2.normalize(msrcr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        # color balance the final MSRCR to flat the histogram distribution with tails on both sides
        msrcr = self.color_balance(msrcr, low_per, high_per)

        return msrcr

    def msrcp(self, img, sigma_scales=[15, 80, 250], low_per=1, high_per=1):
        # Multi-scale retinex with Color Preservation
        # Int(x,y) = sum(Ic(x,y))/3, c={0...k-1}, k=no.of channels
        # MSR_Int(x,y) = MSR(Int(x,y)), and apply color balance
        # B(x,y) = MAX_VALUE/max(Ic(x,y))
        # A(x,y) = max(B(x,y), MSR_Int(x,y)/Int(x,y))
        # MSRCP = A*I

        # Intensity image (Int)
        int_img = (np.sum(img, axis=2) / img.shape[2]) + 1.0
        # Multi-scale retinex of intensity image (MSR)
        msr_int = self.msr(int_img, sigma_scales)
        # color balance of MSR
        msr_cb = self.color_balance(msr_int, low_per, high_per)

        # B = MAX/max(Ic)
        B = 256.0 / (np.max(img, axis=2) + 1.0)
        # BB = stack(B, MSR/Int)
        BB = np.array([B, msr_cb / int_img])
        # A = min(BB)
        A = np.min(BB, axis=0)
        # MSRCP = A*I
        msrcp = np.clip(np.expand_dims(A, 2) * img, 0.0, 255.0)

        return msrcp.astype(np.uint8)

    def apply_gamma_correction(image, gamma=1.2):
        # Build a 256-element lookup table
        look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")

        # Apply the LUT using cv2.LUT
        corrected_image = cv2.LUT(image, look_up_table)

        return corrected_image
