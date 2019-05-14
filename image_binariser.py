import cv2
import numpy as np
import argparse as ap


def get_new_sizes(height, width, new_width):
    new_height = new_width*height/width

    nhd, nhm = divmod(new_height, 4)
    print(nhd, nhm)
    new_height = 4 * (nhd + 1) if (nhd % 2) else 4 * nhd

    return int(new_height), new_width


def make_ridiculous_image(binary, height, width, small):
    new_img = np.zeros((height*8, width*8, 3))
    for i in range(0, height):
        for j in range(0, width):
            col = (int(small[i, j, 0]), int(small[i, j, 1]), int(small[i, j, 2]))
            if binary[i, j] == 0:
                cv2.putText(new_img, "0", (j*8, i*8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, col, 1)
            elif binary[i, j] == 255:
                cv2.putText(new_img, "1", (j*8, i*8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, col, 1)
    return new_img


def main():
    img = cv2.imread(args.image)
    new_width = (args.width//8) * 8
    height, width = get_new_sizes(img.shape[0], img.shape[1], new_width=new_width)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (width, height))
    small = cv2.resize(img, (width, height))
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    new_img = make_ridiculous_image(binary, height, width, small)
    cv2.namedWindow("test")

    while True:
        cv2.imshow("test", new_img)
        ch = cv2.waitKey(1) & 0xFF
        if ch == ord('q'):
            break
    cv2.imwrite("binarised.jpg", new_img)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, default="infinity_knot.jpg",
                        help="path to image to be binarised")
    parser.add_argument('-w', '--width', type=int, default=112,
                        help="width to resize to before replacing with characters")
    args = parser.parse_args()
    main()