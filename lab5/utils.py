import os
import cv2

import numpy as np
from math import floor
from numpy.linalg import svd, inv


def projectImage(frames, sourceFrameIndex, referenceFrameIndex,
                 pastHomographies, originTranslations, xrange=1632,
                 yrange=512, overlapThreshold=40000, errorThreshold=4e-4,
                 numKeyframes=3, checkAllKeyframes=0, auto_H_func=None,
                 homography_func=None, normalization_func=None):
    '''
    Input:
        - frames: 4D array of frames
        - sourceFrameIndex: index of the frame to be projected
        - referenceFrameIndex: index of the frame to be projected to
        - pastHomographies: 2D cell array caching previously computed
          homographies from every frame to every other frame
        - xrange, yrange: dimensions of the output image
        - overlapThreshold: sufficient number of pixels overlapping between
          projected source and reference frames to ensure good homography
        - errorThreshold: acceptable error for good homography
        - numKeyframes: number of equidistant keyframes between source and
          reference frame to be visited in search of better homography
        - checkAllKeyframes: 0 if algorithms breaks after first better
          homography is found, 1 if all keyframes are to be visited

    Output:
        - bestProjectedImage: source frame optimally projected onto reference
          frame using reestimation of homography based on closest-frame search
          and using closest-frame homography as
    '''
    numFrames = frames.shape[0]

    _, referenceTransform, ref_origin_coord = transformImage(frames, referenceFrameIndex, referenceFrameIndex, pastHomographies, xrange, yrange, auto_H_func, homography_func, normalization_func)
    _, sourceTransform, src_origin_coord = transformImage(frames, sourceFrameIndex, referenceFrameIndex, pastHomographies, xrange, yrange, auto_H_func, homography_func, normalization_func)
    _, err = computeOverlap(sourceTransform, src_origin_coord, referenceTransform, ref_origin_coord, overlapThreshold)
    originTranslations[sourceFrameIndex] = src_origin_coord

    x_min, y_min = originTranslations[0]
    # Translation matrix
    t = [-x_min, -y_min]
    H_t = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32)

    # Dot product of translation matrix and homography
    pastHomographies[sourceFrameIndex, referenceFrameIndex] = H_t.dot(pastHomographies[sourceFrameIndex, referenceFrameIndex])

    projectedImage = cv2.warpPerspective((frames[sourceFrameIndex]*255).astype(np.uint8),
                                         pastHomographies[sourceFrameIndex, referenceFrameIndex],
                                         (xrange, yrange))

    if err > errorThreshold:
        print('Finding better homography...')
        increment = floor(((referenceFrameIndex - sourceFrameIndex) - 1) / (numKeyframes + 1))
        keyframeIndex = sourceFrameIndex + increment  # frame being used to find better homography from source to reference
        found = 0
        counter = 0
        bestHomography = np.eye(3)  # initialize H as identity

        while counter < numKeyframes and keyframeIndex < numFrames and keyframeIndex > 0:

            # compute homography and projected image from keyframe to
            # reference frame
            H2, keyframeTransform, keyframe_origin_coord = transformImage(frames, keyframeIndex, referenceFrameIndex, pastHomographies, xrange, yrange, auto_H_func, homography_func, normalization_func)
            a, error1 = computeOverlap(keyframeTransform, keyframe_origin_coord, referenceTransform, ref_origin_coord, overlapThreshold)

            # compute homography and projected image from source frame to
            # keyframe (new reference = keyframe)
            _, keyframeToKeyframeTransform, keyframeToKeyframe_origin_coord = transformImage(frames, keyframeIndex, keyframeIndex, pastHomographies, xrange, yrange,  auto_H_func, homography_func, normalization_func)
            H1, sourceToKeyframeTransform, srcToKeyframe_origin_coord = transformImage(frames, sourceFrameIndex, keyframeIndex, pastHomographies, xrange, yrange,  auto_H_func, homography_func, normalization_func)
            b, error2 = computeOverlap(sourceToKeyframeTransform, srcToKeyframe_origin_coord, keyframeToKeyframeTransform, keyframeToKeyframe_origin_coord, overlapThreshold)

            sufficientOverlap = (a and b)

            if (sufficientOverlap and max(error1, error2) < err):
                found = 1
                bestHomography = np.dot(H1, H2)
                src_origin_coord = keyframe_origin_coord + srcToKeyframe_origin_coord
                if not checkAllKeyframes:
                    break

            keyframeIndex = keyframeIndex + increment
            counter = counter + 1

        if found:
            print('Found better homography')
            pastHomographies[sourceFrameIndex, referenceFrameIndex] = bestHomography
            originTranslations[sourceFrameIndex] = src_origin_coord
            min_origin_coord = np.amin(originTranslations, axis=0)

            x_min, y_min = originTranslations[0]
            # Translation matrix
            t = [-x_min, -y_min]
            H_t = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32)

            # Dot product of translation matrix and homography
            T = H_t.dot(bestHomography)

            projectedImage = cv2.warpPerspective((frames[sourceFrameIndex]*255).astype(np.uint8), T, (xrange, yrange))

            pastHomographies[sourceFrameIndex, referenceFrameIndex] = T.astype(np.float32)

    return projectedImage, pastHomographies, originTranslations


def computeOverlap(sourceTransform, src_origin_coord, referenceTransform,
                   ref_origin_coord, overlapThreshold):
    '''
    Input:
        - sourceTransform: source frame projected onto reference frame plane
        - referenceTransform: reference frame projected onto same space
        - overlapThreshold: sufficient number of pixels overlapping between
          projected source and reference frames to ensure good homography

    Output:
        - sufficientOverlap: boolean indicating whether sufficient overlap is
          found between projected source and reference frame
        - error: raw pixel intensity error between projected source
          and reference frames in the overlapping region
    '''

    img_origins = np.stack((ref_origin_coord, src_origin_coord))
    [x_min, y_min] = np.ceil(img_origins.min(axis=0)).astype(int)

    # Translation matrix
    t = [-x_min, -y_min]
    H_t = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32)

    canvas_dim = (sourceTransform.shape[1], sourceTransform.shape[0])
    referenceTransform = cv2.warpPerspective(referenceTransform, H_t, canvas_dim)

    referenceTransform = (referenceTransform / 255.0).astype(np.float32)
    sourceTransform = (sourceTransform / 255.0).astype(np.float32)

    nnzReference = referenceTransform != 0
    nnzSource = sourceTransform != 0
    nnzCommon = nnzReference * nnzSource

    if not np.sum(nnzCommon):
        error = np.finfo(np.float32).max
    else:
        error = np.sqrt(np.sum((sourceTransform[nnzCommon] - referenceTransform[nnzCommon])**2)) / np.sum(nnzCommon)

    print('Overlap:{}'.format(np.sum(np.prod(nnzCommon, axis=-1))))
    print('Error:{}'.format(error))

    sufficientOverlap = np.sum(np.prod(nnzCommon, axis=-1)) > overlapThreshold

    return sufficientOverlap, error


def transformImage(frames, sourceFrameIndex, referenceFrameIndex,
                   pastHomographies, xrange, yrange, auto_H_func,
                   homography_func, normalization_func):
    '''
    Input:
        - frames: 4D array of frames
        - sourceFrameIndex: index of the frame to be projected
        - referenceFrameIndex: index of the frame to be projected to
        - pastHomographies: 2D cell array caching previously computed
          homographies from every frame to every other frame
        - xrange, yrange: dimensions of the output image

    Output:
        - homography: homography from source frame to reference frame
        - imageTransform: projected source frame with dimensions (xrange, yrange)
    '''

    if pastHomographies[sourceFrameIndex, referenceFrameIndex].any():
        homography = pastHomographies[sourceFrameIndex, referenceFrameIndex]
    else:
        homography = auto_H_func(frames[sourceFrameIndex], frames[referenceFrameIndex],
                                 homography_func, normalization_func)
        pastHomographies[sourceFrameIndex, referenceFrameIndex] = homography.astype(np.float32)

    sourceFrame = frames[sourceFrameIndex]

    sourceFrame_corners = get_img_corners(sourceFrame)
    sourceFrame_corners_tranform_corners = cv2.perspectiveTransform(sourceFrame_corners, homography)
    canvas_corners = np.array([[0., 0.], [xrange, 0.], [yrange, xrange], [0., yrange]])

    corners = np.concatenate((np.squeeze(sourceFrame_corners_tranform_corners, axis=1), canvas_corners))
    [x_min, y_min] = np.ceil(corners.min(axis=0)).astype(int)
    [x_max, y_max] = np.ceil(corners.max(axis=0)).astype(int)

    # Translation matrix
    t = [-x_min, -y_min]
    H_t = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32)

    # Dot product of translation matrix and homography
    T = H_t.dot(homography)

    projectedImage = cv2.warpPerspective((sourceFrame*255).astype(np.uint8), T, (xrange, yrange))

    return homography, projectedImage, np.array([x_min, y_min])


def blendImages(sourceTransform, referenceTransform):
    '''
    Input:
        - sourceTransform: source frame projected onto reference frame plane
        - referenceTransform: reference frame projected onto same space

    Output:
        - blendedOutput: naive blending result from frame stitching
    '''

    blendedOutput = referenceTransform
    indices = referenceTransform == 0
    blendedOutput[indices] = sourceTransform[indices]

    return (blendedOutput / blendedOutput.max() * 255).astype(np.uint8)


def get_img_corners(img):
    '''
    Returns the position of the corners of an color image
    Input:
        img: color image.
    Output:
        corners: image corners (np.array)
    '''	
    corners = np.zeros((4, 1, 2), dtype=np.float32)

    height, width, channels = img.shape
    corners[0] = (0, 0) #top left
    corners[1] = (width, 0) #top right
    corners[2] = (width, height) #bottom right
    corners[3] = (0, height) #bottom left
    
    return corners    


def video2imageFolder(input_file, output_path):
    '''
    Extracts the frames from an input video file
    and saves them as separate frames in an output directory.
    Input:
        input_file: Input video file.
        output_path: Output directorys.
    Output:
        None
    '''

    cap = cv2.VideoCapture()
    cap.open(input_file)

    if not cap.isOpened():
        print("Failed to open input video")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_idx = 0

    while frame_idx < frame_count:
        ret, frame = cap.read()

        if not ret:
            print ("Failed to get the frame {}".format(frameId))
            continue

        out_name = os.path.join(output_path, 'f{:04d}.jpg'.format(frame_idx+1))
        ret = cv2.imwrite(out_name, frame)
        if not ret:
            print ("Failed to write the frame {}".format(frame_idx))
            continue

        frame_idx += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


def imageFolder2mpeg(input_path, output_path='./output_video.mpeg', fps=30.0):
    '''
    Extracts the frames from an input video file
    and saves them as separate frames in an output directory.
    Input:
        input_path: Input video file.
        output_path: Output directorys.
        fps: frames per second (default: 30).
    Output:
        None
    '''

    dir_frames = input_path
    files_info = os.scandir(dir_frames)

    file_names = [f.path for f in files_info if f.name.endswith(".jpg")]
    file_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    frame_Height, frame_Width = cv2.imread(file_names[0]).shape[:2]
    resolution = (frame_Width, frame_Height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    frame_count = len(file_names)

    frame_idx = 0

    while frame_idx < frame_count:


        frame_i = cv2.imread(file_names[frame_idx])
        video_writer.write(frame_i)
        frame_idx += 1

    video_writer.release()