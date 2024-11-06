import os
import scipy
import numpy as np
import sympy as sp
import open3d as o3d
from spatialmath import SE3, SO3

from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

from pointnet2 import *

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, m, centroid

def random_partition(n, n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange( n_data )
    np.random.shuffle(all_idxs)
    return all_idxs[:n], all_idxs[n:]

def ransac(data, model, n, k, t, d, inliers_ratio=0.5, debug=False, return_all=False):
    """
    fit model parameters to data using the RANSAC algorithm

    This implementation written from pseudocode found at
    http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

    {{{
    Given:
        data - a set of observed data points, shape is (n, 2/3)
        model - a model that can be fitted to data points
                model must implement two mehod:
                - def fit(data, ...): given data return parameters of model
                - def get_error(data, model): given data and model prameters return error between data and model
        n - the minimum number of data values required to fit the model
        k - the maximum number of iterations allowed in the algorithm
        t - a threshold value for determining when a data point fits a model
        d - the number of close data values required to assert that a model fits well to data
        inliers_ratio - the number of inliers gt than the raion will return
    Return:
        bestfit - model parameters which best fit the data (or nil if no good model is found)
        inliers_idxs(option) - the index of inliers of the data
    Pesudo code:
        iterations = 0
        bestfit = nil
        besterr = something really large
        while iterations < k {
            maybeinliers = n randomly selected values from data
            maybemodel = model parameters fitted to maybeinliers
            alsoinliers = empty set
            for every point in data not in maybeinliers {
                if point fits maybemodel with an error smaller than t add point to alsoinliers
            }
            if the number of elements in alsoinliers is > d {
                % this implies that we may have found a good model
                % now test how good it is
                bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
                thiserr = a measure of how well model fits these points
                if thiserr < besterr {
                    bestfit = bettermodel
                    besterr = thiserr
                }
            }
            increment iterations
        }
        return bestfit
    }}}
    """
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    data_size = data.shape[0]
    inliers_condition = inliers_ratio * data_size
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data_size)
        maybeinliers = data[maybe_idxs,:]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error(test_points, maybemodel)
        also_idxs = test_idxs[test_err < t] # select indices of rows with accepted points
        alsoinliers = data[also_idxs,:]
        alsoinliers_num = len(alsoinliers)
        if debug:
            print('test_err.min()',test_err.min())
            print('test_err.max()',test_err.max())
            print('np.mean(test_err)',np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d'%(
                iterations,len(alsoinliers)))
        if alsoinliers_num > d:
            betterdata = np.concatenate((maybeinliers, alsoinliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
        iterations += 1
        if alsoinliers_num + n > inliers_condition:
            break
    if bestfit is None:
        print("Fit failed")
    if return_all:
        return bestfit, best_inlier_idxs
    else:
        return bestfit

class CircleLeastSquaresModel:
    """
        data shape is (n, 2)
    """
    def fit(self, data):
        x = data[:, 0]
        y = data[:, 1]
        size = data.shape[0]
        diff_x = 2 * (x[:, np.newaxis] - x[np.newaxis, :])[np.triu_indices(size, k=1)]
        diff_y = 2 * (y[:, np.newaxis] - y[np.newaxis, :])[np.triu_indices(size, k=1)]
        xy_2 = x**2 + y**2
        B = (xy_2[:, np.newaxis] - xy_2[np.newaxis, :])[np.triu_indices(size, k=1)]
        A = np.hstack((diff_x[:, np.newaxis], diff_y[:, np.newaxis]))
        center = np.linalg.lstsq(A, B, rcond=None)[0]
        r = np.mean(np.linalg.norm(np.hstack((x[:, np.newaxis], y[:, np.newaxis])) - center, axis=1))
        return (*center, r)
    def get_error(self, data, model):
        x = data[:, 0]
        y = data[:, 1]
        x0, y0, r = model
        return np.abs((x - x0)**2 + (y - y0)**2 - r**2)

class EllipsoidLeastSquaresModel:
    """
        data shape is (n, 3)
    """
    def fit(self, data):
        A = []
        data_size = len(data)
        if data_size > 100:
            data = data[np.random.choice(data_size, 100, replace=False)]
        for x in data:
            A.append([x[0]**2, x[1]**2, x[2]**2, x[0] * x[1], x[0] * x[2], x[1] * x[2], x[0], x[1], x[2], 1])
        B = np.zeros(data.shape[0])
        U, S, V = scipy.linalg.svd(A)
        model = V[-1]
        if self.get_ellipsoid_params(model) is None:
            return None
        return model
    def get_error(self, data, model):
        if model is None:
            return np.array([np.inf]*data.shape[0])
        A = []
        for x in data:
            A.append([x[0]**2, x[1]**2, x[2]**2, x[0] * x[1], x[0] * x[2], x[1] * x[2], x[0], x[1], x[2], 1])
        return np.abs((np.array(A) @ np.array(model)[:, np.newaxis])[:,0])
    def get_ellipsoid_params(self, model):
        a, b, c, d, e, f, g, h, i, j = model
        x, y, z = sp.symbols('x y z')
        expr = a*x**2 + b*y**2 + c*z**2 + d*x*y + e*x*z + f*y*z + g*x + h*y + i*z + j
        A0 = sp.Matrix([[a, d/2, e/2],
                        [d/2, b, f/2],
                        [e/2, f/2, c]])
        eigenvalues = sorted(A0.eigenvals())
        eigenvectors = sorted(A0.eigenvects(), key=lambda x: abs(x[0]))
        R = eigenvectors[2][2][0].col_insert(0, eigenvectors[1][2][0]).col_insert(0, eigenvectors[0][2][0])
        if np.linalg.det(np.asarray(R, dtype=np.float64)) + 1 < 1e-3:
            return None
        xp, yp, zp = sp.symbols('xp yp zp')
        xyz = sp.Matrix([[xp, yp, zp]]) * R.T
        trans = expr.subs({x: xyz[0], y: xyz[1], z: xyz[2]})
        if np.any(np.array(eigenvalues) < 0):
            trans = trans * (-1)
        var = (xp**2, yp**2, zp**2, xp*yp, xp*zp, yp*zp, xp, yp, zp, 1)
        expr_trans = trans.expand()
        coefficients_dict = expr_trans.as_coefficients_dict(*var)
        for term, coefficient in coefficients_dict.items():
            if abs(coefficient) < 1e-10:
                expr_trans = expr_trans.subs(term, 0)
        if coefficients_dict[1] > 0:
            return None
        coefficients_dict = expr_trans.as_coefficients_dict(*var)
        coeff_xp2 = float(coefficients_dict[xp**2])
        coeff_xp = float(coefficients_dict[xp])
        coeff_yp2 = float(coefficients_dict[yp**2])
        coeff_yp = float(coefficients_dict[yp])
        coeff_zp2 = float(coefficients_dict[zp**2])
        coeff_zp = float(coefficients_dict[zp])

        if coeff_zp2 < 0 or coeff_yp2 < 0 or coeff_xp2 < 0:
            return None

        x0 = -0.5 * coeff_xp / coeff_xp2
        y0 = -0.5 * coeff_yp / coeff_yp2
        z0 = -0.5 * coeff_zp / coeff_zp2
        x0t, y0t, z0t = (sp.Matrix([x0, y0, z0]).T * R.T).tolist()[0]

        Cx = coeff_xp2 * x0**2
        Cy = coeff_yp2 * y0**2
        Cz = coeff_zp2 * z0**2

        constJ = float(abs(coefficients_dict[1] - Cx - Cy - Cz))

        a = np.sqrt(constJ / coeff_xp2)
        b = np.sqrt(constJ / coeff_yp2)
        c = np.sqrt(constJ / coeff_zp2)

        return (x0t, y0t, z0t, a, b, c, R)

class NormalLeastSquaresModel:
    """
        data shape is (n, 3)
    """
    def fit(self, data):
        init_guess = np.array([0.57735027, 0.57735027, 0.57735027])
        data_size = len(data)
        if data_size > 100:
            data = data[np.random.choice(data_size, 100, replace=False)]
        result = scipy.optimize.minimize(self.angle_diff, init_guess, args=(data,)) # 优化搜索使夹角余弦差最小
        vector = result.x / np.linalg.norm(result.x)
        angle = np.mean(np.arccos(np.dot(data, vector)))
        return vector, angle
    def get_error(self, data, model):
        vector, angle = model
        angles = np.arccos(np.dot(data, vector))
        return np.abs(angles - angle)
    def angle_diff(self, X, normals):
        X = X / np.linalg.norm(X)
        size = len(normals)
        cos_theta_list_np = np.dot(normals, X)
        diff_matrix = cos_theta_list_np[:, np.newaxis] - cos_theta_list_np[np.newaxis, :]
        return np.sum(diff_matrix[np.triu_indices(size, k=1)] ** 2)

class ConeAxisLeastSquaresModel:
    """
        data shape is (n, 3)
    """
    def fit(self, data):
        data_size = len(data)
        if data_size > 100:
            data = data[np.random.choice(data_size, 100, replace=False)]
        init_guess = np.array([0.57735027, 0.57735027, 0.57735027]) # [1,1,1] normalized
        result = scipy.optimize.minimize(self.angle_diff_variance, init_guess, args=(data)) # 优化搜索使夹角余弦方差最小
        vector = result.x / np.linalg.norm(result.x)
        angle = np.mean(np.arccos(np.dot(data, vector)))
        return vector, angle
    def get_error(self, data, model):
        vector, angle = model
        angles = np.arccos(np.clip(np.dot(data, vector), -1, 1))
        return np.abs(angles - angle)
    def angle_diff_variance(self, X, normals):
        X = X / np.linalg.norm(X)
        angles = np.arccos(np.clip(np.dot(normals, X), -1, 1))
        # angles = angles[angles > 0.2]
        return np.var(angles)

def fit_frustum_by_slice_poly(points, num_layers=10):
    '''
        Fit frustum cone by slice points along Z-axis
    Params:
    - points: input points
    - num_layers: the number of slice layers
    Return:
    - r1: radius of max Z slice layers
    - r2: radius of min Z slice layers
    - height: max Z - min Z
    - center: center coordinate at X-Y plane [x,y]
    '''
    x, y, z = points[:,0], points[:,1], points[:,2]
    z_min = z.min()
    z_max = z.max()
    height = z_max - z_min
    z_step = height / num_layers
    layer_pts = []
    for i in range(num_layers):
        layer_min = z_min + i * height / num_layers
        layer_max = z_min + (i + 1) * height / num_layers
        mask = (z >= layer_min) & (z < layer_max)
        layer_pts.append(points[mask])
    layer_radius = {}
    layer_center = {}
    for layer_idx in range (num_layers):
        layer_idx_pts = layer_pts[layer_idx]
        cirModel = CircleLeastSquaresModel()
        bestmodel, inliers = ransac(layer_idx_pts[:,:2], cirModel, 3, 100, 0.01, 10, debug=False, return_all=True)
        if bestmodel is not None:
            layer_radius[layer_idx] = bestmodel[2]
            layer_center[layer_idx] = bestmodel[:2]
    X_to_fit = np.array(list(layer_radius.keys()))[:, np.newaxis]
    y_to_fit = np.array(list(layer_radius.values()))
    model = make_pipeline(PolynomialFeatures(degree=2), RANSACRegressor(LinearRegression()))
    model.fit(X_to_fit, y_to_fit.ravel())
    ransacModel = model.named_steps['ransacregressor']
    inlier_mask = ransacModel.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    # inlier_mask = np.where(ransac.inlier_mask_)[0]
    # outlier_mask = np.where(np.logical_not(ransac.inlier_mask_))[0]
    line_x_ransac = np.array(range(num_layers))[:, np.newaxis]
    line_y_ransac = model.predict(line_x_ransac)
    r2 = line_y_ransac[0] # From min to max, so r2 is bottom
    r1 = line_y_ransac[-1]
    layer_center = np.array(list(layer_center.values()))
    center = np.array([np.mean(layer_center[inlier_mask, 0]), np.mean(layer_center[inlier_mask, 1]), (z_min + z_max) / 2])
    return r1, r2, height, center

class ShapeFitter():
    def __init__(self, checkpoint_path, num_class, use_cuda):
        self.use_cuda = use_cuda
        self.num_class = num_class
        self.classifier = get_model(num_class, normal_channel=True)
        if use_cuda:
            self.classifier = self.classifier.cuda()
        classifier_checkpoint = torch.load(checkpoint_path)
        self.classifier.load_state_dict(classifier_checkpoint['model_state_dict'])
        self.classifier.eval()
        print("ShapeFitter Classifier model initialized!")

    def get_pcd_category(self, pcd):
        points = torch.from_numpy(np.asarray(pcd.points))
        normals = torch.from_numpy(np.asarray(pcd.normals))
        ptsWithN = torch.cat((points, normals), dim=1)
        ptsWithNT = torch.unsqueeze(ptsWithN.permute(1, 0), 0)
        vote_pool = torch.zeros(1, self.num_class)
        if self.use_cuda:
            ptsWithNT = ptsWithNT.cuda()
            vote_pool = vote_pool.cuda()
        pred, _ = self.classifier(ptsWithNT.float())
        print(vote_pool)
        print(pred)
        vote_pool += pred
        pred = vote_pool / 1
        pred_choice = pred.data.max(1)[1]
        print(pred)
        print(pred_choice)
        return pred_choice.item()

    def fit_pcd_by_cuboid(self, pcd):
        '''
        Fit cuboid by "Oriented Bounding Box"
        Params:
        - pcd: input point cloud
        Return:
        - a: side length along X axis
        - b: side length along Y axis
        - c: side length along Z axis
        - T: rigid transform of the cuboid
        '''
        obb = pcd.get_minimal_oriented_bounding_box()
        T = SE3.Rt(obb.R, obb.center)
        pcd.transform(T.inv())
        aabb = pcd.get_axis_aligned_bounding_box()
        pcd.transform(T)
        a = aabb.max_bound[0]
        b = aabb.max_bound[1]
        c = aabb.max_bound[2]
        return a, b, c, T

    def fit_pcd_by_frustum(self, pcd):
        '''
        Fit frustum cone by RANSAC method
        Params:
        - pcd: input point cloud
        Return:
        - r1: radius 1
        - r2: radius 2
        - height: height of the frustum cone
        - T: rigid transform of the frustum cone at center
        '''
        pcd_normalized = o3d.geometry.PointCloud(pcd)
        pts, m, centroid = pc_normalize(np.asarray(pcd.points))
        pcd_normalized.points = o3d.utility.Vector3dVector(pts)
        points = np.asarray(pcd_normalized.points)
        normals = np.asarray(pcd_normalized.normals)
        num_points = len(points)
        num_clusters = 10
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, tol = 0.01).fit(normals)
        labels = kmeans.labels_
        cluster_normals_list = []
        plane_normals_list = []
        plane_points_ratio_list = []
        for i in range(num_clusters):
            cluster_indices = np.where(labels == i)[0]
            cluster_indices_size = len(cluster_indices)
            if cluster_indices_size < num_points / 10:
                continue
            cluster_pcd = pcd_normalized.select_by_index(cluster_indices)
            cluster_normal = np.mean(normals[cluster_indices], axis=0)
            cluster_normals_list.append(cluster_normal)
            plane_model, plane_inliers = cluster_pcd.segment_plane(distance_threshold=0.001, ransac_n=3, num_iterations=1000)
            plane_normals_list.append(plane_model[:3])
            ratio = len(plane_inliers) / cluster_indices_size
            plane_points_ratio_list.append(ratio)
            # print(f"idx: {i}, plane_normals_list: {plane_model[:3]}, cluster_normal: {cluster_normal}")
            # print(f"pts: {cluster_indices_size}, Ratio: {ratio}")
            # plane_pcd = cluster_pcd.select_by_index(plane_inliers)
            # plane_pcd.paint_uniform_color([1,0,0])
            # o3d.visualization.draw_geometries([cluster_pcd, plane_pcd], window_name='Classified Point Clouds', point_show_normal=True)
        print(plane_points_ratio_list)
        if np.max(plane_points_ratio_list) > 0.7 and len([x for x in plane_points_ratio_list if x > 0.7]) < 3:
            print(f"使用平面法向量")
            max_plane_ratio_idx = np.argmax(plane_points_ratio_list)
            cone_normal = plane_normals_list[max_plane_ratio_idx]
        else:
            print(f"使用估计法向量")
            cone_axis_model = ConeAxisLeastSquaresModel()
            cluster_normals_list = np.array(cluster_normals_list)
            best_fit, _ = ransac(normals, cone_axis_model, 3, 200, 0.02, 1, inliers_ratio=0.99, debug=False, return_all=True)
            vector, best_angle = best_fit
            cone_normal = vector
        vec_x = np.cross(cone_normal, [0,0,1])
        R = SO3.TwoVectors(x=vec_x, z=cone_normal)
        pcd_normalized.rotate(R.inv(), center=[0,0,0])
        """ Fit cone radius  """
        r1, r2, height, center = fit_frustum_by_slice_poly(points, 30)
        return r1 * m, r2 * m, height * m, SE3(centroid) * SE3(R) * SE3(np.asarray(center) * m)

    def fit_pcd_by_ellipsoid(self, pcd):
        points, m, centroid = pc_normalize(np.asarray(pcd.points))
        num_points = len(points)
        ellipsoid_model = EllipsoidLeastSquaresModel()
        best_fit, _ = ransac(points, ellipsoid_model, 10, 200, 0.02, 1, inliers_ratio=0.99, debug=False, return_all=True)
        x0t, y0t, z0t, a, b, c, R = ellipsoid_model.get_ellipsoid_params(best_fit)
        center = np.array([x0t, y0t, z0t]) * m + centroid
        T = SE3.Rt(SO3(np.array(R, dtype=np.float64)), center)
        return a*m, b*m, c*m, T


if __name__ == "__main__":
    model_path = "../../../data/models"
    shape_fitter = ShapeFitter(os.path.join(model_path, "best_model_ssg.pth"), 3, True)
    # classifier_model_type = "pointnet2_cls_ssg"
    # num_class = 3
    # use_normals = True