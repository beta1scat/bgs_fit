import numpy as np
import sympy as sp
import scipy

np.set_printoptions(suppress=True)
def generate_cone_points(r_bottom=10, r_top_ratio=0.5, height=20, delta=0.1, points_density=1, total_points=10000):
    assert r_bottom > 0, "cone r_top should > 0"
    assert height > 0, "cone height should > 0"
    assert points_density >= 0, "number of points density should >= 0"
    assert total_points > 0, "number of points should > 0"
    r_top = r_bottom * r_top_ratio
    half_height = height / 2
    points = []
    area1 = np.pi * r_top * r_top
    area2 = np.pi * r_bottom * r_bottom
    area3 = np.pi * (r_top + r_bottom) * np.sqrt((r_bottom - r_top)**2 + height**2)
    total_area = area1 + area2 + area3
    l = np.sqrt((r_bottom - r_top)**2 + height**2)
    num_points_l = 0
    if points_density != 0:
        num_points_l = int(np.pi * (r_top + r_bottom) * l * points_density)
    else:
        num_points_l = int(total_points * (area3 / total_area))
    for _ in range(num_points_l):
        ratio = np.random.uniform(-1, 1)
        ratio_0_1 = (ratio + 1) / 2
        z = ratio * half_height
        r = ratio_0_1 * (r_top - r_bottom) + r_bottom
        phi = 2 * np.pi * np.random.rand()
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x + np.random.uniform(-1, 1) * delta, y + np.random.uniform(-1, 1) * delta, z + np.random.uniform(-1, 1) * delta])
    return np.array(points)

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

def random_partition(n, n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange( n_data )
    np.random.shuffle(all_idxs)
    return all_idxs[:n], all_idxs[n:]

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

class EllipseLeastSquaresModel:
    """
        data shape is (n, 2)
    """
    def fit(self, data):
        A = []
        for x in data:
            A.append([x[0]**2, x[0]*x[1], x[1]**2, x[0], x[1], 1])
        B = np.zeros(data.shape[0])
        _, _, V = np.linalg.svd(A)
        model = V[-1]
        if self.get_ellipse_params(model) is None:
            return None
        return model
    def get_error(self, data, model):
        if model is None:
            return np.array([np.inf]*data.shape[0])
        A = []
        for x in data:
            A.append([x[0]**2, x[0]*x[1], x[1]**2, x[0], x[1], 1])
        return np.abs((np.array(A) @ np.array(model)[:, np.newaxis])[:,0])
    def get_ellipse_params(self, model):
        a, b, c, d, e, f = model
        b2m4ac = b**2 - 4*a*c
        xc = (2*c*d - b*e) / b2m4ac
        yc = (2*a*e - b*d) / b2m4ac
        center = [xc, yc]
        t = atan(b / (c - a)) / (-2)
        tmpEq = a*xc**2 + b*xc*yc + c*yc**2 - f
        a_square = tmpEq / (a*cos(t)**2 + b*sin(t)*cos(t) + c*sin(t)**2)
        b_square = tmpEq / (c*cos(t)**2 - b*sin(t)*cos(t) + a*sin(t)**2)
        if (a_square < 0 or b_square < 0):
            return None
        a = sqrt(a_square)
        b = sqrt(b_square)
        return (xc, yc, a, b, t)

class EllipsoidLeastSquaresModel:
    """
        data shape is (n, 3)
    """
    def fit(self, data):
        A = []
        for x in data:
            A.append([x[0]**2, x[1]**2, x[2]**2, x[0] * x[1], x[0] * x[2], x[1] * x[2], x[0], x[1], x[2], 1])
        B = np.zeros(data.shape[0])
        start_time = time.time()  # Record start time
        print(f"B.shape: {B.shape}")
        U, S, V = scipy.linalg.svd(A)
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"SVD time: {elapsed_time} seconds")
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
        R = eigenvectors[0][2][0].T.row_insert(0, eigenvectors[1][2][0].T).row_insert(0, eigenvectors[2][2][0].T) # Mehtod in book is row insert
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
        coeff_xp2 = expr_trans.coeff(xp**2)
        coeff_xp = expr_trans.coeff(xp)
        coeff_yp2 = expr_trans.coeff(yp**2)
        coeff_yp = expr_trans.coeff(yp)
        coeff_zp2 = expr_trans.coeff(zp**2)
        coeff_zp = expr_trans.coeff(zp)

        x0 = -0.5 * coeff_xp / coeff_xp2
        y0 = -0.5 * coeff_yp / coeff_yp2
        z0 = -0.5 * coeff_zp / coeff_zp2
        x0t, y0t, z0t = (sp.Matrix([x0, y0, z0]).T * R.T).tolist()[0]

        Cx = coeff_xp2 * x0**2
        Cy = coeff_yp2 * y0**2
        Cz = coeff_zp2 * z0**2

        constJ = abs(coefficients_dict[1] - Cx - Cy - Cz)
        a = sqrt(constJ / coeff_xp2)
        b = sqrt(constJ / coeff_yp2)
        c = sqrt(constJ / coeff_zp2)

        return (x0t, y0t, z0t, a, b, c, R)

class NormalLeastSquaresModel:
    """
        data shape is (n, 3)
    """
    def fit(self, data):
        init_guess = np.array([0.0, 0.0, 1.0])
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
        init_guess = np.random.rand(3)
        result = scipy.optimize.minimize(self.angle_diff_variance, init_guess, args=(data)) # 优化搜索使夹角余弦方差最小
        print(result)
        vector = result.x / np.linalg.norm(result.x)
        angle = np.max(np.arccos(np.dot(data, vector)))
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
