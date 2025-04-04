import gc
import multiprocessing
import os.path
import sys
import time
import dlib
from multiprocessing.managers import BaseManager
from PIL import Image
from deepface.commons.functions import load_image
from deepface.detectors import FaceDetector
from tqdm import tqdm
import cv2
import numpy as np
import torch
from deepface import DeepFace
import face_recognition
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def rearrange_profile_pic(image):
    image = face_recognition.load_image_file(image)
    pad_size = ((30, 30), (30, 30), (0, 0))

    image = np.pad(image, pad_size, mode='constant', constant_values=0).astype(np.uint8)
    # Find facial landmarks using face_recognition
    face_landmarks_list = face_recognition.face_landmarks(image)

    if len(face_landmarks_list) == 0:
        print("No faces found in the image.")
        return None
    else:
        # Take the first detected face (you can iterate over multiple faces)
        face_landmarks = face_landmarks_list[0]
        height, width, channels = image.shape
        # Extract the (x, y) coordinates of the nose tip
        nose_x = face_landmarks['nose_tip'][0][0]
        nose_y = face_landmarks['nose_tip'][0][1]

        # Calculate the desired center of the image
        desired_center_x = image.shape[1] // 2
        desired_center_y = image.shape[0] // 2

        # Calculate the translation needed to center the nose
        x_offset = desired_center_x - nose_x
        y_offset = desired_center_y - nose_y + 40

        # Create an empty canvas with the same size as the input image
        centered_image = np.zeros_like(image)

        # Calculate the region of interest for the input image within the canvas
        input_roi_x1 = max(-x_offset, 0)
        input_roi_x2 = min(-x_offset + width, width)
        input_roi_y1 = max(-y_offset, 0)
        input_roi_y2 = min(-y_offset + height, height)

        # Copy the input image to the canvas
        centered_image[max(0, y_offset):min(image.shape[0], image.shape[0] + y_offset),
        max(0, x_offset):min(image.shape[1], image.shape[1] + x_offset)] = image[input_roi_y1:input_roi_y2,
                                                                           input_roi_x1:input_roi_x2]

        center_x = centered_image.shape[1] // 2
        center_y = centered_image.shape[0] // 2
        if centered_image.shape[0] < centered_image.shape[1]:
            centered_image = centered_image[:, center_x - center_y:center_x + center_y]
        if centered_image.shape[0] > centered_image.shape[1]:
            centered_image = centered_image[center_y - center_x:center_y + center_x, :]

    return cv2.resize(centered_image, (112, 112))


def get_files_full_path(rootdir):
    import os
    paths = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                paths.append(os.path.join(root, file))
    return paths


def resume(imgs, cropped):
    import os
    basepath = "/".join(imgs[0][::-1].split("/")[3:])[::-1]
    relatives, images = [], []
    for crp in cropped:
        relatives.append("/".join(crp.split("/")[-3::]))
    for img in imgs:
        images.append("/".join(img.split("/")[-3::]))

    diff = list(set(images).difference(relatives))

    for i, d in enumerate(diff):
        diff[i] = os.path.join(basepath, d)
    return diff


def exclude(imgs,  exclude_file):
    files = []
    with open(exclude_file, "r") as ex:
        for r in ex:
            files.append(r.strip())
    diff = list(set(imgs).difference(files))
    diff_n = len(imgs) - len(diff)
    print("Excluded %d files" % diff_n)
    return diff


def detect(face_detector, backend, img, target_size=(112, 112), align=True, grayscale=False, enforce_detection=True):
    try:
        img = load_image(img)
        """
        if img.shape[0] > img.shape[1] and img.shape[0] > 512:
            ratio = img.shape[0] / 512
            width = int(img.shape[1] / ratio)
            height = int(img.shape[0] / ratio)
            dim = (width, height)
            img = cv2.resize(img, dim)
        if img.shape[0] < img.shape[1] and img.shape[1] > 512:
            ratio = img.shape[1] / 512
            width = int(img.shape[1] / ratio)
            height = int(img.shape[0] / ratio)
            dim = (width, height)
            img = cv2.resize(img, dim)
        """
        base_img = img.copy()
        img, img_region = FaceDetector.detect_face(face_detector, backend, img, align)
        # --------------------------

        if img.shape[0] == 0 or img.shape[1] == 0:
            if enforce_detection == True:
                raise ValueError("Detected face shape is ", img.shape,
                                 ". Consider to set enforce_detection argument to False.")
            else:  # restore base image
                img = base_img.copy()

        # --------------------------

        # post-processing
        if grayscale == True:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ---------------------------------------------------
        # resize image to expected shape

        # img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image

        if img.shape[0] > 0 and img.shape[1] > 0:
            factor_0 = target_size[0] / img.shape[0]
            factor_1 = target_size[1] / img.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
            img = cv2.resize(img, dsize)

            # Then pad the other side to the target size by adding black pixels
            diff_0 = target_size[0] - img.shape[0]
            diff_1 = target_size[1] - img.shape[1]
            if grayscale == False:
                # Put the base image in the middle of the padded image
                img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),
                             'constant')
            else:
                img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)),
                             'constant')

        # ------------------------------------------

        # double check: if target image is not still the same size with target.
        if img.shape[0:2] != target_size:
            img = cv2.resize(img, target_size)

        # ---------------------------------------------------

        # normalizing the image pixels

        #img_pixels = image.img_to_array(img)  # what this line doing? must?
        img_pixels = np.expand_dims(img, axis=0)
        #img_pixels = np.divide(img_pixels, 255, out=img_pixels, casting="unsafe")#/= int(255)  # normalize input in [0, 1]
        return img_pixels[0, :, :, ::-1]
    # ---------------------------------------------------
    except:  # if detected face shape is (0, 0) and alignment cannot be performed, this block will be run
        return None

def preprocess(images, resizeonly, proc_id):
    with tqdm(total=len(images), desc="Process %d" % proc_id) as pbar:
        backends = ['opencv', 'ssd', 'dlib', 'mtcnn']
        b_id = 3
        #face_detector = FaceDetector.build_model(backends[b_id])
        #predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        for image in images:
            start_time = time.time()


            try:
                outfilepath = image.split("/")
                outfilepath[-4] = "Aligned_" + outfilepath[-4]
                sub_root_folder = "/".join(outfilepath[:-1])
                outfilepath = "/".join(outfilepath)

                if not os.path.exists(outfilepath):
                    if not resizeonly:

                        if "profile" in image:
                            cropped_image = rearrange_profile_pic(image)
                        else:
                            cropped_image = DeepFace.detectFace(image, detector_backend=backends[b_id], target_size=(112, 112), enforce_detection=False, align=True)


                        if cropped_image is None:
                            img = Image.open(image)

                            # Get the dimensions of the image
                            width, height = img.size

                            # Determine the side length for the square image
                            side_length = max(width, height)

                            # Create a new square canvas with a black background
                            square_img = Image.new('RGB', (side_length, side_length), (0, 0, 0))

                            # Calculate the position to paste the original image
                            paste_x = (side_length - width) // 2
                            paste_y = (side_length - height) // 2

                            # Paste the original image onto the square canvas
                            square_img.paste(img, (paste_x, paste_y))

                            # Resize the square image to 112x112
                            square_img = square_img.resize((112, 112), Image.Resampling.LANCZOS)
                            cropped_image = np.asarray(square_img, dtype=np.uint8)


                        #det = dlib.rectangle(int(- 56), int(-56), int(56), int(56))
                        #cropped_image = detect(face_detector, backends[b_id], image)
                    else:
                        cropped_image = cv2.resize(cv2.imread(image), (36, 36))



                    if not os.path.exists(sub_root_folder):
                        os.makedirs(sub_root_folder)
                        #print("Creted folder: %s " % sub_root_folder)
                    #cv2.imwrite(outfilepath, cv2.normalize(cropped_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    cv2.imwrite(outfilepath, cv2.normalize(cropped_image[:, :, ::-1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    gc.collect()
                    #print("Processing time: " + str(time.time() - start_time))
            except Exception as e:
                print(e)
                #pass
                #print("Invalid Image: " + image)
            pbar.update(1)


if __name__ == '__main__':
    import argparse

    try:
        parser = argparse.ArgumentParser(description='Image aligner and cropper')
        parser.add_argument('--imgs', metavar='path', required=True,
                            help='path to images')
        parser.add_argument('--processes', required=True, default=6,
                            help='number of cpu processes')
        parser.add_argument('--resizeonly', required=False, default=False,
                            help='set as True if you just want to resize yoor images')
        parser.add_argument('--exclude', required=False, default=None,
                            help='list of files to exclude')
        args = parser.parse_args()
        if args.imgs == "":
            print("Invalid image path")
    except Exception as e:
        print(e)
        sys.exit(1)
    images = get_files_full_path(args.imgs)
    outfilepath = images[0].split("/")
    outfilepath[-4] = "Aligned_"+outfilepath[-4]
    outfilepath = "/".join(outfilepath[:-3])
    done = get_files_full_path(outfilepath)
    """
    subfolds = [x for x in os.listdir(args.imgs) if x.startswith("Aligned")]
    [done.extend(get_files_full_path(os.path.join(args.imgs, el))) for el in subfolds]
    """
    if len(done) > 0:
        images = resume(images, done)
        print("Already done: %d images - remaining: %d images" % (len(done), len(images)))
    if args.exclude is not None:
        pre = len(images)
        images = exclude(images, args.exclude)
        print("Excluded: %d images - remaining: %d images" % (pre, len(images)))
    num_processes = int(args.processes)
    chunk_length = int(len(images) / num_processes)
    image_chunks = []
    for rank in range(num_processes):
        if rank != num_processes - 1:
            image_chunks.append(images[rank * chunk_length:(rank + 1) * chunk_length])
        else:
            image_chunks.append(images[rank * chunk_length:])
    processes = []
    for i, c in enumerate(image_chunks):
        processes.append(
            multiprocessing.Process(target=preprocess, args=(c, bool(args.resizeonly), i)))

    for t in processes:
        t.start()
    for t in processes:
        t.join()
