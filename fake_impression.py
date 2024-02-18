import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv
import cv2
import random

# A. Rendering strokes
def rendering_strokes(canvas, sizeIm):
    line_length = 4
    angle = 135
    theta = math.radians(angle)
    thickness = 2
    delta = np.array([math.cos(theta), math.sin(theta)])
    # loop until there are no neon green space
    while((57 in canvas[:,:,0]) and (255 in canvas[:,:,1]) and (20 in canvas[:,:,2]) ):
        # Randomly select stroke center
        center = np.floor(np.random.rand(2,1).flatten() * np.array([sizeIm[1],sizeIm[0]])) + 1
        center = np.amin(np.vstack((center, np.array([sizeIm[1], sizeIm[0]]))), axis=0)
        # Grab colour from image at center position of the stroke
        color = np.reshape(im1[int(center[1]-1), int(center[0]-1), :],(3,1))
        color = (int(color[0]), int(color[1]), int(color[2]))
        # Add the stroke to the canvas
        #nx, ny = (sizeIm[1], sizeIm[0])
        start_point = center - delta * line_length 
        start = (int(start_point[0]), int(start_point[1]))
        end_point = center + delta * line_length
        end = (int(end_point[0]), int(end_point[1]))
        canvas = cv2.line(canvas, start, end, color, thickness)

    canvas[canvas < 0] = 0.0
    return canvas

# B. Random Perturbations
def random_perturbations(canvas, sizeIm):
    # loop until there are no neon green space
    while((57 in canvas[:,:,0]) and (255 in canvas[:,:,1]) and (20 in canvas[:,:,2]) ):
        line_length = random.randint(1,5)
        angle = random.randint(90, 180)
        theta = math.radians(angle)
        thickness = 2
        delta = np.array([math.cos(theta), math.sin(theta)])
        # Randomly select stroke center
        center = np.floor(np.random.rand(2,1).flatten() * np.array([sizeIm[1],sizeIm[0]])) + 1
        center = np.amin(np.vstack((center, np.array([sizeIm[1], sizeIm[0]]))), axis=0)
        # Grab colour from image at center position of the stroke
        color = np.reshape(im1[int(center[1]-1), int(center[0]-1), :],(3,1))
        color_r = int(color[0])
        color_g = int(color[1])
        color_b = int(color[2])
        color = (random.randint(color_r-8,color_r+8), random.randint(color_g-8, color_g+8), random.randint(color_b-8, color_b+8))
        # Add the stroke to the canvas
        #nx, ny = (sizeIm[1], sizeIm[0])
        start_point = center - delta * line_length 
        start = (int(start_point[0]), int(start_point[1]))
        end_point = center + delta * line_length
        end = (int(end_point[0]), int(end_point[1]))
        canvas = cv2.line(canvas, start, end, color, thickness)

    
    canvas[canvas < 0] = 0.0
    return canvas

# C. Edge Clipping
def edge_clipping(canvas, sizeIm, grad, theta, value):

    for i in range(value * 20):
        # Randomly select stroke center
        center = np.floor(np.random.rand(2,1).flatten() * np.array([sizeIm[1],sizeIm[0]])) + 1
        center = np.amin(np.vstack((center, np.array([sizeIm[1], sizeIm[0]]))), axis=0)
        # The center of the stroke is given by (cx,cy) and the direction of the stroke is given by (dirx, diry).
        # This process determines (x1,y1) and (x2,y2), the endpoints of the stroke clipped to edges in the image.
        cx = int(center[0])
        cy = int(center[1])

        color = np.reshape(im1[int(center[1]-1), int(center[0]-1), :],(3,1))
        color_r = int(color[0])
        color_g = int(color[1])
        color_b = int(color[2])
        color = (random.randint(color_r-8,color_r+8), random.randint(color_g-8, color_g+8), random.randint(color_b-8, color_b+8))
        thickness = 2

        thetac = theta[cx][cy]
        delta = np.array([math.cos(thetac), math.sin(thetac)])
        dirx = delta[0]
        diry = delta[1]
        length = 4
        step = 0
        
        # C. Clipping and Rendering
        # The Sobel filtered intensity images is sampled in steps of unit length in order to detect edges
        # To determine (x1, y1):
        # a. set (x1, y1) to (cx, cy)
        (x1, y1) = (cx, cy)
        # b. bilinearly sample the Sobel filltered intensity image at (x1,y1), and set lastSample to this value
        lastSample = grad[x1][y1]
        # c. set (tempx, tempy) to (x1+drix, y1+diry), taking a unit step in the orientation direction
        while(step < length):
            (tempx, tempy) = (x1+dirx, y1+diry)
            # d. if (dist((x1,y1),(tempx,tempy)) > (length of stroke / 2) then stop
            if(math.dist((x1, y1),(tempx, tempy)) > (length / 2)):
                break
            # e. bilinearly sample the sobel image at (tempx, tempy), and set newSample to this value
            newSample = grad[int(tempx)][int(tempy)]
            # f. if (newSample < lastSample) then stop
            if(newSample < lastSample):
                break
            # g. set (x1,y1) = (tempx, tempy)
            (x1, y1) = (tempx, tempy)
            # h. set lastSample to newSample
            lastSample = newSample
            # i. go to step c
            step += 1
        
        step = 0
        start = (int(x1), int(y1))
        # To determine (x2, y2): the endpoint in the other direction, set (dirx, diry) to (-dirx, -diry) and repeat the above process
        dirx = -dirx
        diry = -diry
        # a. set (x2, y2) to (cx, cy)
        (x2, y2) = (cx, cy)
        # b. bilinearly sample the Sobel filltered intensity image at (x2,y2), and set lastSample to this value
        lastSample = grad[x2][y2]
        # c. set (tempx, tempy) to (x2+drix, y2+diry), taking a unit step in the orientation direction
        while(step < length):
            (tempx, tempy) = (x2+dirx, y2+diry)
            # d. if (dist((x2,y2),(tempx,tempy)) > (length of stroke / 2) then stop
            if(math.dist((x2, y2),(tempx, tempy)) > (length / 2)):
                break
            # e. bilinearly sample the sobel image at (tempx, tempy), and set newSample to this value
            newSample = grad[int(tempx)][int(tempy)]
            # f. if (newSample < lastSample) then stop
            if(newSample < lastSample):
                break
            # g. set (x2,y2) = (tempx, tempy)
            (x2, y2) = (tempx, tempy)
            # h. set lastSample to newSample
            lastSample = newSample
            # i. go to step c
            step += 1
        end = (int(x2), int(y2))
        
        canvas = cv2.line(canvas, start, end, color, thickness)

    canvas[canvas < 0] = 0.0

    return canvas

# generate the Gaussian 5x5 kernel
def gaussian_kernel(size, sigma=1):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)
    return kernel


# Gradient Calculation
def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = conv(img, Kx)
    Iy = conv(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    # D. Brush Stroke Orientation
    theta = np.arctan2(Iy, Ix)
    theta = math.radians(90) + theta

    return (G, theta)

# FUNCTION USED TO DISPLAY CANNY EDGE DETECTION IMAGE
# Non-Maximum Suppression
def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255

                #angle 0
                if(0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                
                #angle 45
                elif(22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]

                #angle 90
                elif(67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                
                #angle 135
                elif(112.5 <= angle[i,j] < 157.7):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]
                
                if(img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0
            except IndexError as e:
                pass
    return Z

# FUNCTION USED TO DISPLAY CANNY EDGE DETECTION IMAGE
# Double threshold
def threshold(img, lowthresholdratio=0.05, highthresholdratio=0.09):
    highthreshold = img.max() * highthresholdratio
    lowthreshold = highthreshold * lowthresholdratio

    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highthreshold)
    zeros_i, zeros_j = np.where(img < lowthreshold)

    weak_i, weak_j = np.where((img < highthreshold) & (img >= lowthreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

# FUNCTION USED TO DISPLAY CANNY EDGE DETECTION IMAGE
def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if(img[i,j] == weak):
                try:
                    if((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def canny_detector(img):
    # 2. Apply Gaussian Blur
    img_filtered = conv(img, gaussian_kernel(5, sigma=2.0))
    # 3. Apply Sobel filter
    grad, theta = sobel_filters(img_filtered)
    # img_nms = non_max_suppression(grad, theta)
    # img_thresh, weak, strong = threshold(img_nms, lowThresholdRatio=0.07, highThresholdRatio=0.19)
    # img_final = hysteresis(img_thresh, weak, strong=strong)
    return grad, theta


if __name__ == '__main__':
    # Setting up the input output paths
    imageDir = '../Images/'
    outputDir = '../Results/' 

    im_name = 'bridge.jpg'

    im1 = plt.imread(imageDir + im_name)

    width, height, colorvalue = im1.shape
    value = width * height
    sizeIm = im1.shape
    sizeIm = sizeIm[0:2]

    canvas = np.zeros([sizeIm[0], sizeIm[1], 3], dtype=np.uint8)
    canvas.fill(-1)
    # canvas[:,:,0] = 57
    # canvas[:,:,1] = 255
    # canvas[:,:,2] = 20

    # A. Rendering Strokes
    # final_img = rendering_strokes(canvas, sizeIm)
    # B. Random Perturbations
    # final_img = random_perturbations(canvas, sizeIm)

    # C. Edge Clipping and Orientation
    # Create intensity image
    intimg = im1[:,:,0] * 0.30 + im1[:,:,1] * 0.59 + im1[:,:,2] * 0.11
    grad, theta = canny_detector(intimg)
    final_img = edge_clipping(canvas, sizeIm, grad, theta, value)

    plt.imsave(outputDir + 'canny_edge_final.jpg', final_img)