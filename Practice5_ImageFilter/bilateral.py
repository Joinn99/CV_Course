import cv2
import numpy as np
import matplotlib.pyplot as plt


def bilaterl(image, ker_w=3, sigma_d=75.0, sigma_r=75.0):
    mat_dist = np.power(np.meshgrid(np.arange(-ker_w, ker_w + 1),
                                    np.arange(-ker_w, ker_w + 1), indexing='xy'), 2)
    ker_d = - np.sum(mat_dist, axis=0) / (2 * sigma_d**2)

    dst = np.zeros(image.shape)
    for ind_i in range(image.shape[0]):
        for ind_j in range(image.shape[1]):
            i_min = np.maximum(ind_i - ker_w, 0)
            i_max = np.minimum(ind_i + ker_w, image.shape[0] - 1)
            j_min = np.maximum(ind_j - ker_w, 0)
            j_max = np.minimum(ind_j + ker_w, image.shape[1] - 1)
            region = image[i_min:i_max+1, j_min:j_max+1]
            ker_r = - \
                (np.power(region - image[ind_i, ind_j], 2) / (2 * sigma_r**2))
            kernel = np.exp(ker_r + ker_d[(i_min - ind_i + ker_w):(
                i_max - ind_i + ker_w + 1), (j_min - ind_j + ker_w):(j_max - ind_j + ker_w + 1)])
            dst[ind_i, ind_j] = np.sum(np.multiply(
                region, kernel), axis=None) / np.sum(kernel, axis=None)
        print('\rProgress: {:4d} '.format(ind_i + 1) + ' /{:4d}'.format(image.shape[0]), end='')
    return dst


if __name__ == "__main__":
    print('Bilateral filters')
    IMG = cv2.imread('Practice5_ImageFilter/face.jpg', 0)
    DST = bilaterl(IMG)
    FIG, AXES = plt.subplots(1, 3)
    AXES[0].imshow(IMG, 'gray')
    AXES[1].imshow(DST, 'gray')
    AXES[2].imshow(cv2.bilateralFilter(IMG, 15, 75, 75), 'gray')
    AXES[0].set_title('Original')
    AXES[1].set_title('Our Filter')
    AXES[2].set_title('OpenCV Filter')
    FIG.suptitle("Bilateral filters", fontsize=16)
    plt.show()
