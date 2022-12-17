"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        # return homography
        N = match_p_src.shape[1]
        u = match_p_dst[0, :]
        v = match_p_dst[1, :]
        # move to homogeneous coordinates
        xy = np.transpose(np.vstack([match_p_src, np.ones(N)]))

        uxy = np.multiply(np.transpose(np.array([u, ] * 3)), xy)
        vxy = np.multiply(np.transpose(np.array([v, ] * 3)), xy)
        A_u = np.concatenate((-xy, np.zeros((N, 3)), uxy), axis=1)
        A_v = np.concatenate((np.zeros((N, 3)), -xy, vxy), axis=1)
        A = np.zeros((2 * N, 9))
        for i in range(0, N):
            A[2 * i] = A_u[i]
            A[2 * i + 1] = A_v[i]

        AtA = np.dot(np.transpose(A), A)
        _, v = np.linalg.eigh(AtA)
        H = v[:, 0]
        H = np.reshape(H, (3, 3))
        H = H / H[-1, -1]
        return H

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        #TODO types
        def get_new_xy(homograpy: np.ndarray, src_x, src_y):
            dst_u_v = homography.dot(np.array([src_x, src_y, 1]))
            dst_u_v = dst_u_v / dst_u_v[-1]
            return round(dst_u_v[0]), round(dst_u_v[1]) 
            
        dst_image = np.zeros(dst_image_shape, dtype=np.uint8)
        for i in range(len(src_image)):
            for j in range(len(src_image[0])):
                y, x = get_new_xy(homography, j, i)
                if x >=0 and x < dst_image_shape[0] and y >=0 and y < dst_image_shape[1]:
                    dst_image[x][y] = src_image[i][j]
        return dst_image

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        src_image_shape = src_image.shape
        H = src_image_shape[0]
        W = src_image_shape[1]
        dst_image = np.zeros(dst_image_shape, dtype=np.uint8)
        temp = np.full((H), 1)
        mesh = np.array([
            np.concatenate([i*temp for i in range(W)]),        # [0,0,0,0,....1,1,1,1,...W-1,W-1,W-1,W-1]
            np.concatenate([range(H) for i in range(W)]),      # [0,1,2,3,...H-1,0,1,2,3...H-1,0,1,2,3,...H-1]
            np.full((H*W), 1)])                                # [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        dst_mesh = homography.dot(mesh)
        dst_mesh = dst_mesh / dst_mesh[-1]
        dst_mesh = (np.rint(dst_mesh)).astype(int)
        for i in range(H):
            for j in range(W):
                y, x, _ = dst_mesh[:, H*j+i]
                if x >=0 and x < dst_image_shape[0] and y >=0 and y < dst_image_shape[1]:
                    dst_image[x][y] = src_image[i][j]
        return dst_image
        
    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        N = len(match_p_src[0])
        h_uv_src = np.vstack([match_p_src, np.ones(N)])
        h_uv_dst = homography.dot(h_uv_src)
        h_uv_dst = h_uv_dst / h_uv_dst[-1]
        h_uv_dst = (np.rint(h_uv_dst)).astype(int)
        num_inliers = 0
        sum_square_errors = 0
        for i in range(N):
            if (abs(match_p_dst[0][i] - h_uv_dst[0][i]) + abs(match_p_dst[1][i] - h_uv_dst[1][i]) <= max_err):
                num_inliers += 1
                sum_square_errors += (abs(match_p_dst[0][i] - h_uv_dst[0][i]) + abs(match_p_dst[1][i] - h_uv_dst[1][i]))**2
        if num_inliers == 0:
            return 0, 10**9
        return num_inliers / N, sum_square_errors / num_inliers

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        N = len(match_p_src[0])
        h_uv_src = np.vstack([match_p_src, np.ones(N)])
        h_uv_dst = homography.dot(h_uv_src)
        h_uv_dst = h_uv_dst / h_uv_dst[-1]
        h_uv_dst = (np.rint(h_uv_dst)).astype(int)
        mp_src_meets_model = np.zeros((2,N), dtype=np.int)
        mp_dst_meets_model = np.zeros((2,N), dtype=np.int)
        inliers_found = 0
        for i in range(N):
            if (abs(match_p_dst[0][i] - h_uv_dst[0][i]) + abs(match_p_dst[1][i] - h_uv_dst[1][i]) <= max_err):
                mp_src_meets_model[:, inliers_found] = match_p_src[:, i]
                mp_dst_meets_model[:, inliers_found] = match_p_dst[:, i]
                inliers_found += 1
        return mp_src_meets_model[:, 0:inliers_found], mp_dst_meets_model[:, 0:inliers_found]
        
    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # # use class notations:
        w = inliers_percent
        t = max_err
        # # p = parameter determining the probability of the algorithm to
        # # succeed
        p = 0.99
        # # the minimal probability of points which meets with the model
        d = 0.5
        # # number of points sufficient to compute the model
        n = 4
        # # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        
        N = match_p_src.shape[1]
        min_dist_mse = 10**9
        best_homography = None
        for i in range(k):
            indices = sample(range(N), n)
            subset_match_p_src = np.take(match_p_src, indices, axis=1)
            subset_match_p_dst = np.take(match_p_dst, indices, axis=1)
            homography = Solution.compute_homography_naive(subset_match_p_src, subset_match_p_dst)
            src_inliers, dst_inliers = Solution.meet_the_model_points(homography, match_p_src, match_p_dst, t)
            if ((1.0*src_inliers.shape[1])/N) >= d:
                homography = Solution.compute_homography_naive(src_inliers, dst_inliers)
                _, dist_mse = Solution.test_homography(homography, src_inliers, dst_inliers, t)
                if dist_mse < min_dist_mse:
                    min_dist_mse = dist_mse
                    best_homography = homography
        return best_homography

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        # return backward_warp
        x = np.linspace(0, dst_image_shape[1]-1, dst_image_shape[1])
        y = np.linspace(0, dst_image_shape[0]-1, dst_image_shape[0])
 
        xi, yi = np.meshgrid(x.astype(np.int32), y.astype(np.int32), indexing='ij')
        xi_flattern = xi.flatten()
        yi_flattern = yi.flatten()
        ones_param = np.ones_like(xi_flattern)

        dest_mesh = np.array([xi_flattern, yi_flattern, ones_param])
        dest_h = backward_projective_homography @ dest_mesh
        dest_c = dest_h / dest_h[2]


        x_src = np.linspace(0, src_image.shape[1]-1, src_image.shape[1])
        y_src = np.linspace(0, src_image.shape[0]-1, src_image.shape[0])
 
        xi_src, yi_src = np.meshgrid(x_src.astype(np.int32), y_src.astype(np.int32), indexing='ij')
        xi_src_flattern = xi_src.flatten()
        yi_src_flattern = yi_src.flatten()

        src_mesh = np.array([xi_src_flattern, yi_src_flattern], dtype=np.int32)
        src_values = src_image.T.reshape(3, -1)

        inter_src = griddata(src_mesh.T, src_values.T, dest_c[:2].T, method='cubic', fill_value=0)
        ret_image = inter_src.reshape((dst_image_shape[0], dst_image_shape[1],3), order='F')
        return np.array(ret_image, np.int32)


    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        dx = -pad_left
        dy = -pad_up
        translation = np.identity(3)
        translation[0][-1] = dx
        translation[1][-1] = dy
        final_homography = backward_homography.dot(translation)
        return final_homography
        
    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        ransac_homography = Solution.compute_homography(self, match_p_src,
                                                    match_p_dst,
                                                    inliers_percent,
                                                    max_err)
        print("panorama ransac", ransac_homography)
        panorama_rows_num, panorama_cols_num, pad = Solution.find_panorama_shape(
                                                    src_image,
                                                    dst_image,
                                                    ransac_homography)
        backward_homography = Solution.compute_homography(self, match_p_dst,
                                                    match_p_src,
                                                    inliers_percent,
                                                    max_err)
        print(backward_homography @ np.array((1,1,1)))
        trans_backwards_homography = Solution.add_translation_to_backward_homography(
                                                    backward_homography,
                                                    pad.pad_left,
                                                    pad.pad_up)
        print(backward_homography @ np.array((1+pad.pad_up,1+pad.pad_left,1)))
        backward_mapping = Solution.compute_backward_mapping(trans_backwards_homography,
                                                    src_image,
                                                    (panorama_rows_num, panorama_cols_num, 3))
        panorama_shape = (dst_image.shape[0]+pad.pad_up+pad.pad_down+1, dst_image.shape[1]+pad.pad_left+pad.pad_right+1, 3)
        img_panorama = np.zeros(panorama_shape, dtype=np.uint8)
  
        # embed src image in panorama
        for i in range(panorama_rows_num):
            for j in range(panorama_cols_num):
                img_panorama[i][j] = backward_mapping[i][j]
        # embed dst image in panorama
        img_panorama[pad.pad_up:pad.pad_up+dst_image.shape[0], pad.pad_left:pad.pad_left+dst_image.shape[1],:] = dst_image
        return img_panorama
