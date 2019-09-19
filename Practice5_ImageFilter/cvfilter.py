import cv2
import matplotlib.pyplot as plt


def cvfilter():
    img = cv2.imread('Practice5_ImageFilter/face.jpg', 0)
    blur = cv2.blur(img, (5, 5))
    median_blur = cv2.medianBlur(img, 5)
    gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(img, 'gray')
    axes[0, 1].imshow(blur, 'gray')
    axes[1, 0].imshow(median_blur, 'gray')
    axes[1, 1].imshow(gaussian_blur, 'gray')

    axes[0, 0].set_title('Original')
    axes[0, 1].set_title('Mean Filter')
    axes[1, 0].set_title('Median Filter')
    axes[1, 1].set_title('Gaussian Filter')
    fig.suptitle("Image filters", fontsize=16)

    plt.show()

if __name__ == "__main__":
    cvfilter()
