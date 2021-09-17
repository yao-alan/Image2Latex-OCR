import os
import re
import bisect
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from vocab import Vocab

def data_to_df(dir, max_entries=None, ftype="png"):
    """Creates pandas dataframe of (index | image_path | type | split)."""
    regex_phrases = ['.*test', '.*train', '.*validate']

    if not os.path.exists(dir):
        print("Error: path doesn't exist.")
        return

    files = [f for f in os.listdir(dir) 
                if os.path.isfile(os.path.join(dir, f))]

    files, folders = [], []

    for f in os.listdir(dir):
        pathname = os.path.join(dir, f)
        if os.path.isfile(pathname):
            files.append(f)
        elif os.path.isdir(pathname):
            folders.append(f)

    df = pd.DataFrame(columns=['index', 'image_path', 'type', 'split'])

    img_folder_path = None
    for f in folders:
        match = re.search('.*images', f)
        if match is not None:
            img_folder_path = match.group(0)

    if img_folder_path is None:
        print("Error: no folder contains the name 'images'. If a folder \
               does contain images, please rename it.")

    for r in regex_phrases:
        for f in files:
            match = re.search(r, f)
            if match is not None:
                with open(f'{dir}/{match.group(0)}.lst') as data:
                    lines = data.readlines()

                print(f'{dir}/{match.group(0)}.lst')
                total_num = len(lines) if max_entries is None else max_entries

                for i, l in enumerate(lines[:max_entries]):
                    s = l.split(' ')
                    s[2] = s[2].strip('\n')
                    df = df.append({
                        'index': int(s[0]), 
                        'image_path': f"{dir}/{img_folder_path}/{s[1]}.{ftype}",
                        'type': s[2], 
                        'split': r[2:]},
                        ignore_index=True)

                    if i % 100 == 0:
                        clear_output(wait=True)
                        print(f"{i+1}/{total_num}")

    df = df.sort_values("index").reset_index(drop=True)

    return df


def extract_labels(path, encoding='ISO-8859-1'):
    """Extract ground truth labels for each image."""
    with open(path, newline='\n', encoding=encoding) as f:
        labels = f.readlines()

    return pd.DataFrame(labels, columns=["label"])


def tokenize(label):
    """Split LaTeX sequence into individual tokens."""
    def group(char):
        if char is None:
            return None
        if char == '\\':
            return 0
        elif char.isalpha():
            return 1
        elif char.isdecimal():
            return 2
        else:
            return 3

    split_label = ""
    command = False
    for i in range(len(label)):
        # if label[i] == ' ':
        #     continue 
        
        prv = label[i-1] if i != 0 else None
        cur = label[i]
        nxt = label[i+1] if i != len(label)-1 else None

        g_p, g_c, g_n = group(prv), group(cur), group(nxt)
        if g_c == 0: # on '\'
            command = True
            if g_p is None:
                split_label += '\\'
            else:
                if g_p == 0:
                    if split_label[-1] == ' ':
                        split_label += '\\'
                    else: # literal backslash; not beginning a command
                        split_label += '\\ '
                        command = False
                else:
                    split_label += '\\'
        elif g_c != 3:
            if g_c == g_n and command:
                split_label += cur
            else:
                split_label += f'{cur} '
                command = False
        else:
            split_label += f'{cur} '
            command = False

    return split_label.split()


def reshape_images(img_path, aspect_ratios=None, image_sizes=None, reshape_strat='pad'):
    """Scale images to proper shape."""
    _, img = crop_equations(img_path)
        
    # figure out best padding
    pad_height, pad_width = None, None
    if image_sizes is not None:
        # pad with 8 px to all sides
        img = np.pad(img, pad_width=8, mode='constant', constant_values=255)

        # scale to certain aspect ratio
        if reshape_strat == 'scale':
            # look for best aspect ratio
            # second element of tuple is irrelevant
            curr_ratio = img.shape[1] / img.shape[0]
            idx = bisect.bisect_left(aspect_ratios, (curr_ratio, -1))
            reshape_to = None
            if idx == 0:
                reshape_to = 0
            elif idx == len(aspect_ratios):
                reshape_to = len(aspect_ratios)-1
            else:
                if curr_ratio / aspect_ratios[idx-1] <= aspect_ratios[idx] / curr_ratio:
                    reshape_to = idx-1
                else:
                    reshape_to = idx
            img = cv2.resize(
                img, 
                dsize=(aspect_ratios[reshape_to][1], aspect_ratios[reshape_to][0]),
                interpolation=cv2.INTER_AREA
            )
            pad_height = aspect_ratios[reshape_to][0]
            pad_width = aspect_ratios[reshape_to][1]
        # pad to fit image size
        elif reshape_strat == 'pad':
            img = cv2.resize(
                img, 
                dsize=(
                    min(int(img.shape[1]/2), 320),
                    min(int(img.shape[0]/2), 50)
                ),
                interpolation=cv2.INTER_AREA
            )
            # plt.imshow(img)
            # plt.show()
            # figure out which shape to pad to
            pad_dh, pad_dw = np.inf, np.inf
            for _, (height, width) in enumerate(image_sizes):
                dh = height-img.shape[0]
                dw = width-img.shape[1]
                if 0 <= dh and dh <= pad_dh and 0 <= dw and dw <= pad_dw:
                    pad_dh = height-img.shape[0]
                    pad_dw = width-img.shape[1]
                    pad_height = height
                    pad_width = width
            # if odd padding, top padded less than bottom and left less than right
            top = int(pad_dh/2)
            left = int(pad_dw/2)
            img = np.pad(
                img,
                pad_width=((top, pad_dh-top), (left, pad_dw-left)),
                mode='constant',
                constant_values=255
            )

    img = np.abs(255-img)
    img = np.expand_dims(img, axis=(0))

    return img, pad_height, pad_width


# image_sizes is a list of tuples(height, width) to try to fit to (if batching)
# reshape_strat = 'scale', 'pad', 'None'
def scale_images(merge_shuffle_df, maxlen=None, image_sizes=None, 
                 reshape_strat='pad', model=None):
    """Scale images to proper shape and create proper labels."""
    vocab = Vocab()
    images = {}
    # Harvard's image sizes:
    #   (40, 160), (40, 200), (40, 240), (40, 280), (40, 320)
    #   (50, 120), (50, 200), (50, 240), (50, 280)
    aspect_ratios = []
    seq_len_dict = {}
    if image_sizes is not None:
        for i, (height, width) in enumerate(image_sizes):
            aspect_ratios.append((width / height, i))
            if maxlen is None:
                assert model is not None
                seq_len_dict[(height, width)] = model.get_seq_len(height, width)
        aspect_ratios.sort()

    for i, row in merge_shuffle_df.iterrows():
        # remove \label{...} and tokenize
        #label = tokenize(re.sub(r'\\label\{.*\}', '', row['label']))
        label = tokenize(row['label'])
        # print(label)
        vocab.update(label)

        img, pad_height, pad_width = scale_images(
            row['image_path'], aspect_ratios, image_sizes, reshape_strat
        )

        merge_shuffle_df.at[i, 'label'] = label
        merge_shuffle_df.at[i, 'padded_height'] = pad_height
        merge_shuffle_df.at[i, 'padded_width'] = pad_width
        images[(row['index'], row['dataset'])] = img
        
        print(f'{i+1}/{len(merge_shuffle_df)}')
        clear_output(wait=True)
        
    for i, row in merge_shuffle_df.iterrows():
        if maxlen is None:
            maxlen = None if image_sizes is None \
                else seq_len_dict[(row['padded_height'], row['padded_width'])]
        merge_shuffle_df.at[i, 'label_token_indices'] = np.array(
            vocab.label_to_index(row['label'], vocab, maxlen=maxlen)
        )
        
        print(f'{i+1}/{len(merge_shuffle_df)}')
        clear_output(wait=True)

    return merge_shuffle_df, images, vocab


def crop_equations(path, show_image=False):
    """Simplistic method to isolate the singular equation in the image."""
    img = cv2.imread(path, 0)

    LR_mask = cv2.reduce(img, 0, cv2.REDUCE_MIN) != 255
    UD_mask = cv2.reduce(img, 1, cv2.REDUCE_MIN) != 255
    c1, c2 = LR_mask.argmax(), LR_mask.shape[1] - np.flip(LR_mask).argmax()
    r1, r2 = UD_mask.argmax(), UD_mask.shape[0] - np.flip(UD_mask).argmax()

    if show_image:
        plt.imshow(img, cmap='gray')
        plt.show()
        
    return path, img[r1:r2, c1:c2]


# parameters to tune: Canny lower threshold, HoughLinesP threshold, 
#   kernel, erosion/dilation iterations
def crop_equations_advanced(path, h_smear=80, w_smear=2, iter=5, show_image=False):
    """Extracts bounding boxes for equations; to be used if image is more \
        complex."""
    img = cv2.imread(path, 0)

    # Otsu thresholding to create binary image
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, edges = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Canny edge detection + Hough transform to try to clean up horizontal
    #   and vertical lines (say, from binder paper)
    # can loop multiple times, but seems like once is best
    # HoughLines taken from OpenCV docs example
    for i in range(1):
        edges = cv2.Canny(edges, 75, 150)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 500, None, 0, 0)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(edges, pt1, pt2, (0, 0, 0), 3, cv2.LINE_AA)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, None, 25, 10)
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(edges, (l[0], l[1]), (l[2], l[3]), (0, 0, 0), 3, cv2.LINE_AA)

    # pick a more horizontal kernel
    kernel = np.ones((1, 60))
    edges = cv2.erode(cv2.dilate(edges, kernel, iterations=5), kernel, iterations=8)

    if show_image:
        plt.figure(figsize=(12, 8))
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.subplot(122), plt.imshow(255 - edges, cmap='gray')
        plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_smear, w_smear))
    dilated = cv2.dilate(edges, kernel, iterations=5)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)

    img_bounds = []
    for i in range(len(contours)-1, -1, -1):
        x, y, w, h = cv2.boundingRect(contours[i])
        # to limit false positives, only append those images with a 
        #   brightness percentage above a certain threshold
        if np.sum(255 - img[y:y+h, x:x+w]) / (w * h) >= 20:
            img_bounds.append((y, y+h, x, x+w))

    return (path, img_bounds)
