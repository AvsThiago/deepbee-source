import numpy as np
import cv2


LEFT_BAR_SIZE = 480


def extract_cells_images(image, cells, output_size=224, 
                         mean_radius_default=32):
    cells[:, 2] = cells[:, 2] // 2
    
    # use the mean radius to calculate the clip size to each detection
    # we are standardising the images based on the radius here
    cells[:, 2] = output_size / mean_radius_default * cells[:, 2]
    # the border needs to be greater than the biggest clip
    size_border = cells[:, 2].max() + 1
    # move the detection centers
    cells[:, [0, 1]] = cells[:, [0, 1]] + size_border

    # creates a border around the main image
    img_w_border = cv2.copyMakeBorder(
        image,
        size_border,
        size_border,
        size_border,
        size_border,
        cv2.BORDER_REFLECT,
    )

    # extracts all detections and resizes them
    ROIs = [
        cv2.resize(
            img_w_border[i[1] - i[2] : i[1] + i[2], i[0] - i[2] : i[0] + i[2]],
            (output_size, output_size),
        )
        for i in cells
    ]
    
    blob_imgs = np.asarray([i for i in ROIs])
    return blob_imgs


def draw_labels_bar(image, labels, colors):
    height = image.shape[0]
    left_panel = np.zeros((height, LEFT_BAR_SIZE, 3), dtype=np.uint8)
    labels = [l.title() for l in labels]

    for i, cl in enumerate(zip(colors, labels)):
        color, label = cl
        cv2.putText(
            left_panel,
            " ".join([str(i + 1), ".", label]),
            (15, 70 * (i + 1)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.4,
            color,
            2,
        )

    return np.hstack((left_panel, image))


def draw_circles_labels(image, labels, points, colors=None, draw_labels=True):
    if colors is None:

        colors = [
            (255, 0, 0),
            (0, 255, 255),
            (0, 0, 128),
            (255, 0, 255),
            (0, 255, 0),
            (255, 255, 100),
            (0, 0, 255),
        ]

    if draw_labels:
        image = draw_labels_bar(np.copy(image), labels, colors)

    points[:, 0] += LEFT_BAR_SIZE

    for p in points:
        cv2.circle(image, (p[0], p[1]), p[2], colors[p[3]], 4)

    points[:, 0] -= LEFT_BAR_SIZE
    return image

    