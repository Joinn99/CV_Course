import cv2
import matplotlib.pyplot as plt


def feature_match(data='bikes', image=2, detector=cv2.BRISK_create(), normtype=cv2.NORM_L2):

    img1 = cv2.imread(
        'Practice7_FeatureExtract/Mikolajczyk/{}/img1.ppm'.format(data))
    img2 = cv2.imread(
        'Practice7_FeatureExtract/Mikolajczyk/{}/img{}.ppm'.format(data, image))

    # find the keypoints and descriptors with SIFT
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # create BFMatcher object
    bf_matcher = cv2.BFMatcher(normtype, crossCheck=True)

    # Match descriptors.
    matches = bf_matcher.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    return cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=2)


def method_test(data='bikes', image=2):
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(feature_match(data, image, cv2.ORB_create(), cv2.NORM_HAMMING))
    axes[0, 1].imshow(feature_match(data, image, cv2.BRISK_create(), cv2.NORM_HAMMING))
    axes[1, 0].imshow(feature_match(data, image, cv2.xfeatures2d.SIFT_create()))
    axes[1, 1].imshow(feature_match(data, image, cv2.xfeatures2d.SURF_create()))
    axes[0, 0].set_title('ORB')
    axes[0, 1].set_title('BRISK')
    axes[1, 0].set_title('SIFT')
    axes[1, 1].set_title('SURF')
    fig.suptitle("Feature detect methods comparision", fontsize=16)
    plt.show()


if __name__ == "__main__":
    method_test('graf', 2)
