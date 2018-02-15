import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy
from numpy.linalg import svd
from skimage import data, img_as_float
from skimage import measure
from skimage.color import rgb2gray


def print_result_decorator(original_function):
    def wrapper(*args, **kwargs):
        res = original_function(*args, **kwargs)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
        plt.gray()

        ax[0].imshow(args[0])
        ax[0].set_title('Original Image')

        ax[1].imshow(res)
        ax[1].set_title('Result')
        plt.show()

    return wrapper


def perform_svd(matrix, k):
    """
    Performs SVD on two dimantional matrix
    :param m: matrix
    :param k: number of singular vectors to be used during the matrix reconstruction
    :return: a matrix reconstructed using k singular vectors
    """
    U, s, V = svd(matrix)
    return np.matrix(U[:, :k]) * np.diag(s[:k]) * np.matrix(V[:k, :])


def perform_percent_svd(matrix, percent=0.9):
    """
    Performs SVD on two dimensional matrix
    
    :param matrix: matrix
    :param percent: the compression rate 
    :return: reconstructed matrix 
    """
    U, s, V = svd(matrix)
    total_energy = np.sum(s)
    k = 1
    while (np.sum(s[:k]) / total_energy) <= percent:
        k += 1
    return np.matrix(U[:, :k]) * np.diag(s[:k]) * np.matrix(V[:k, :])


#@print_result_decorator
def compress_image_with_svd(img_to_compress, k):
    """
    :param img_to_compress: 
    :param k: 
    :return: 
    """
    img_matrix = img_as_float(rgb2gray(img_to_compress))
    compressed = perform_svd(img_matrix, k)
    return compressed


#@print_result_decorator
def perform_dwt(image_to_compress):
    #scipy.misc.imsave('photo/real.jpg', image_to_compress)
    coefs = pywt.dwt2(image_to_compress, 'haar')
    compressed, (a, b, c) = coefs
    #scipy.misc.imsave('photo/compressed.jpg', compressed)
    return compressed


if __name__ == '__main__':
    img_matrix = img_as_float(rgb2gray(data.astronaut()))
    # svd_compressed = compress_image_with_svd(data.astronaut(), 50)

    svd_compressed = perform_percent_svd(img_matrix)

    fully_compressed = perform_dwt(svd_compressed)

    svd_compression_ratio = measure.compare_psnr(img_matrix, svd_compressed)
    dwt_compression_ratio = measure.compare_psnr(svd_compressed, fully_compressed)
    total_compression_ratio = measure.compare_psnr(img_matrix, fully_compressed)

    print(svd_compression_ratio)
    print(dwt_compression_ratio)
    print(total_compression_ratio)
