import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy
import sys
from io import StringIO
from io import BytesIO
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
    return np.array(np.matrix(U[:, :k]) * np.diag(s[:k]) * np.matrix(V[:k, :]))


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
    return compressed, coefs


def get_size(obj, seen=None):
    """Recursively finds size of objects
    Looks not ok for our case"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


if __name__ == '__main__':
    img_matrix = img_as_float(rgb2gray(data.astronaut()))
    # svd_compressed = compress_image_with_svd(data.astronaut(), 50)

    svd_compressed = perform_percent_svd(img_matrix)

    fully_compressed, dwt_decomposition = perform_dwt(svd_compressed)

    dwt_restored = pywt.idwt2(dwt_decomposition, 'haar')

    psnr_svd = measure.compare_psnr(img_matrix, svd_compressed)
    #psnr_dwt = measure.compare_psnr(svd_compressed, dwt_restored)
    psnr_total = measure.compare_psnr(img_matrix, dwt_restored)

    print(psnr_svd)
    #print(psnr_dwt)
    print(psnr_total)

    #-WTF?
    print('Original: ' + str(get_size(img_matrix)))
    print('SVD Compressed: ' + str(get_size(svd_compressed)))
    print('DWT Compressed: ' + str(get_size(fully_compressed)))
    print('DWT Decomposition: ' + str(get_size(dwt_decomposition)))  # ??? Where is the compression? Need to find other way for calculations
    print('Restored: ' + str(get_size(dwt_restored)))

    import pickle

    orig = pickle.dumps(img_matrix)
    get_size(pickle)
