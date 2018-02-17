import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy
import os
import sys
from io import StringIO
from io import BytesIO
from numpy.linalg import svd
from skimage import data, img_as_float
from skimage import measure
from skimage.color import rgb2gray

BASE_PROCESSING_FOLDER = 'images/processing'


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
    return np.array(compressed), coefs


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


def calc_comp_rate(original, compressed):
    return os.path.getsize(original) / os.path.getsize(compressed)


def run_pipeline(im_pathes, include_svd = True, include_dwt = False, svd_percentage = 0.9, save_temporary_results = False):
    if not include_svd and not include_dwt:
        raise ValueError('Do at least something!')
    results = []

    for im_path in im_pathes:
        im_name = im_path.split('/')[-1]
        original_matrix = img_as_float(rgb2gray(scipy.misc.imread(im_path)))
        compressed = original_matrix
        orig_path = os.path.join(BASE_PROCESSING_FOLDER, '{}.png'.format(im_name))
        scipy.misc.imsave(orig_path, original_matrix)
        res = {'picture':im_path}

        if include_svd:
            compressed = perform_percent_svd(compressed, svd_percentage)
            svd_im_path = os.path.join(BASE_PROCESSING_FOLDER,
                                           'svd_c_{}.png'.format(im_name))
            scipy.misc.imsave(svd_im_path, compressed)
            res['svd_compression'] = calc_comp_rate(orig_path, svd_im_path)
            res['svd_psnr'] = measure.compare_psnr(original_matrix, compressed)

        if include_dwt:
            compressed, coefs = perform_dwt(compressed)
            dwt_im_path = os.path.join(BASE_PROCESSING_FOLDER,
                                       'dwt_c_{}.png'.format(im_name))

            #some issue with this part
            scipy.misc.imsave(dwt_im_path, compressed)
            res['dwt_decomposed_compression'] = calc_comp_rate(orig_path, dwt_im_path)

            compressed = pywt.idwt2(coefs, 'haar')
            restored_im_path = os.path.join(BASE_PROCESSING_FOLDER,
                                       'restored_c_{}.png'.format(im_name))
            scipy.misc.imsave(restored_im_path, compressed)
            res['dwt_restored_compression'] = calc_comp_rate(orig_path, restored_im_path)
            res['dwt_restored_psnr'] = measure.compare_psnr(original_matrix, compressed)

        results.append(res)

    return results







if __name__ == '__main__':
    # img_matrix = img_as_float(rgb2gray(data.astronaut()))
    # # svd_compressed = compress_image_with_svd(data.astronaut(), 50)
    #
    # svd_compressed = perform_percent_svd(img_matrix)
    #
    # fully_compressed, dwt_decomposition = perform_dwt(svd_compressed)
    #
    # dwt_restored = pywt.idwt2(dwt_decomposition, 'haar')
    #
    # psnr_svd = measure.compare_psnr(img_matrix, svd_compressed)
    # #psnr_dwt = measure.compare_psnr(svd_compressed, dwt_restored)
    # psnr_total = measure.compare_psnr(img_matrix, dwt_restored)
    #
    # print(psnr_svd)
    # #print(psnr_dwt)
    # print(psnr_total)
    #
    # #-WTF?
    # print('Original: ' + str(get_size(img_matrix)))
    # print('SVD Compressed: ' + str(get_size(svd_compressed)))
    # print('DWT Compressed: ' + str(get_size(fully_compressed)))
    # print('DWT Decomposition: ' + str(get_size(dwt_decomposition)))  # ??? Where is the compression? Need to find other way for calculations
    # print('Restored: ' + str(get_size(dwt_restored)))
    #
    # import pickle
    #
    # orig = pickle.dumps(img_matrix)
    #get_size(pickle)


    print(run_pipeline(['images/real/boat.png'], include_dwt=True))
