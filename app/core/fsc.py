import numpy as np


def lsm(match1, match2, change_form):
    match1_xy = match1[:, [0, 1]]
    match2_xy = match2[:, [0, 1]]
    A_rows = []
    for i in range(match1_xy.shape[0]):
        x, y = match1_xy[i]
        A_rows.append([x, y, 0.0, 0.0, 1.0, 0.0])
        A_rows.append([0.0, 0.0, x, y, 0.0, 1.0])
    A = np.array(A_rows, dtype=float)
    b = match2_xy.T.reshape(-1, order="F")
    if change_form == "affine":
        Q, R = np.linalg.qr(A)
        parameters = np.linalg.solve(R, Q.T @ b)
        parameters = np.concatenate([parameters, [0.0, 0.0]])
        N = match1.shape[0]
        match1_test = match1_xy
        match2_test = match2_xy
        M = np.array([[parameters[0], parameters[1]], [parameters[2], parameters[3]]], dtype=float)
        match1_test_trans = (M @ match1_test.T) + np.array([[parameters[4]], [parameters[5]]])
        match1_test_trans = match1_test_trans.T
        test = match1_test_trans - match2_test
        rmse = np.sqrt(np.sum(test ** 2) / N)
    elif change_form == "perspective":
        temp_1_rows = []
        for i in range(match1_xy.shape[0]):
            x, y = match1_xy[i]
            temp_1_rows.append([-x, -y])
            temp_1_rows.append([-x, -y])
        temp_1 = np.array(temp_1_rows, dtype=float)
        temp_2 = np.repeat(b[:, None], 2, axis=1)
        temp = temp_1 * temp_2
        A = np.concatenate([A, temp], axis=1)
        Q, R = np.linalg.qr(A)
        parameters = np.linalg.solve(R, Q.T @ b)
        N = match1.shape[0]
        match1_test = match1_xy.T
        match1_test = np.vstack([match1_test, np.ones((1, N))])
        M = np.array(
            [
                [parameters[0], parameters[1], parameters[4]],
                [parameters[2], parameters[3], parameters[5]],
                [parameters[6], parameters[7], 1.0],
            ],
            dtype=float,
        )
        match1_test_trans = M @ match1_test
        match1_test_trans_12 = match1_test_trans[:2, :]
        match1_test_trans_3 = np.repeat(match1_test_trans[2:3, :], 2, axis=0)
        match1_test_trans = match1_test_trans_12 / match1_test_trans_3
        match1_test_trans = match1_test_trans.T
        match2_test = match2_xy
        test = match1_test_trans - match2_test
        rmse = np.sqrt(np.sum(test ** 2) / N)
    elif change_form == "similarity":
        A_rows = []
        for i in range(match1_xy.shape[0]):
            x, y = match1_xy[i]
            A_rows.append([x, y, 1.0, 0.0])
            A_rows.append([y, -x, 0.0, 1.0])
        A = np.array(A_rows, dtype=float)
        Q, R = np.linalg.qr(A)
        parameters = np.linalg.solve(R, Q.T @ b)
        parameters = np.concatenate([parameters, [0.0, 0.0, 0.0, 0.0]])
        parameters[4:6] = parameters[2:4]
        parameters[2] = -parameters[1]
        parameters[3] = parameters[0]
        N = match1.shape[0]
        match1_test = match1_xy
        match2_test = match2_xy
        M = np.array([[parameters[0], parameters[1]], [parameters[2], parameters[3]]], dtype=float)
        match1_test_trans = (M @ match1_test.T) + np.array([[parameters[4]], [parameters[5]]])
        match1_test_trans = match1_test_trans.T
        test = match1_test_trans - match2_test
        rmse = np.sqrt(np.sum(test ** 2) / N)
    else:
        raise ValueError("Unsupported change_form")
    return parameters, rmse


def _unique_rows_first_indices(values):
    seen = {}
    indices = []
    for idx, row in enumerate(values):
        key = (row[0], row[1])
        if key not in seen:
            seen[key] = idx
            indices.append(idx)
    return np.array(sorted(indices), dtype=int)


class _RandStream:
    def __init__(self, values):
        self.values = np.array(values, dtype=float).reshape(-1)
        self.index = 0

    def random(self, size):
        count = int(np.prod(size))
        if self.index + count > self.values.size:
            raise ValueError("RandStream exhausted")
        out = self.values[self.index : self.index + count]
        self.index += count
        return out.reshape(size)


def fsc(cor1, cor2, change_form, error_t, rng=None):
    cor1 = np.array(cor1, dtype=float)
    cor2 = np.array(cor2, dtype=float)
    M, N = cor1.shape
    if change_form == "similarity":
        n = 2
        max_iteration = M * (M - 1) / 2
    elif change_form == "affine":
        n = 3
        max_iteration = M * (M - 1) * (M - 2) / (2 * 3)
    elif change_form == "perspective":
        n = 4
        max_iteration = M * (M - 1) * (M - 2) / (2 * 3)
    else:
        raise ValueError("Unsupported change_form")
    iterations = 10000 if max_iteration > 10000 else int(max_iteration)
    most_consensus_number = 0
    cor1_new = np.zeros((M, N), dtype=float)
    cor2_new = np.zeros((M, N), dtype=float)
    if rng is None:
        rng = np.random.RandomState()
    for _ in range(iterations):
        while True:
            a = np.floor(1 + (M - 1) * rng.random((1, n))).astype(int).flatten()
            cor11 = cor1[a - 1, :2]
            cor22 = cor2[a - 1, :2]
            if n == 2:
                if (
                    a[0] != a[1]
                    and np.any(cor11[0, :2] != cor11[1, :2])
                    and np.any(cor22[0, :2] != cor22[1, :2])
                ):
                    break
            if n == 3:
                if (
                    a[0] != a[1]
                    and a[0] != a[2]
                    and a[1] != a[2]
                    and np.any(cor11[0, :2] != cor11[1, :2])
                    and np.any(cor11[0, :2] != cor11[2, :2])
                    and np.any(cor11[1, :2] != cor11[2, :2])
                    and np.any(cor22[0, :2] != cor22[1, :2])
                    and np.any(cor22[0, :2] != cor22[2, :2])
                    and np.any(cor22[1, :2] != cor22[2, :2])
                ):
                    break
            if n == 4:
                if (
                    a[0] != a[1]
                    and a[0] != a[2]
                    and a[0] != a[3]
                    and a[1] != a[2]
                    and a[1] != a[3]
                    and a[2] != a[3]
                    and np.any(cor11[0, :2] != cor11[1, :2])
                    and np.any(cor11[0, :2] != cor11[2, :2])
                    and np.any(cor11[0, :2] != cor11[3, :2])
                    and np.any(cor11[1, :2] != cor11[2, :2])
                    and np.any(cor11[1, :2] != cor11[3, :2])
                    and np.any(cor11[2, :2] != cor11[3, :2])
                    and np.any(cor22[0, :2] != cor22[1, :2])
                    and np.any(cor22[0, :2] != cor22[2, :2])
                    and np.any(cor22[0, :2] != cor22[3, :2])
                    and np.any(cor22[1, :2] != cor22[2, :2])
                    and np.any(cor22[1, :2] != cor22[3, :2])
                    and np.any(cor22[2, :2] != cor22[3, :2])
                ):
                    break
        parameters, _ = lsm(cor11, cor22, change_form)
        solution = np.array(
            [
                [parameters[0], parameters[1], parameters[4]],
                [parameters[2], parameters[3], parameters[5]],
                [parameters[6], parameters[7], 1.0],
            ],
            dtype=float,
        )
        match1_xy = cor1[:, :2].T
        match1_xy = np.vstack([match1_xy, np.ones((1, M))])
        if change_form == "perspective":
            match1_test_trans = solution @ match1_xy
            match1_test_trans_12 = match1_test_trans[:2, :]
            match1_test_trans_3 = np.repeat(match1_test_trans[2:3, :], 2, axis=0)
            match1_test_trans = match1_test_trans_12 / match1_test_trans_3
            match1_test_trans = match1_test_trans.T
            match2_xy = cor2[:, :2]
            test = match1_test_trans - match2_xy
            diff_match2_xy = np.sqrt(np.sum(test ** 2, axis=1))
            index_in = np.where(diff_match2_xy < error_t)[0]
            consensus_num = index_in.size
        else:
            t_match1_xy = solution @ match1_xy
            match2_xy = cor2[:, :2].T
            match2_xy = np.vstack([match2_xy, np.ones((1, M))])
            diff_match2_xy = t_match1_xy - match2_xy
            diff_match2_xy = np.sqrt(np.sum(diff_match2_xy ** 2, axis=0))
            index_in = np.where(diff_match2_xy < error_t)[0]
            consensus_num = index_in.size
        if consensus_num > most_consensus_number:
            most_consensus_number = consensus_num
            cor1_new = cor1[index_in, :]
            cor2_new = cor2[index_in, :]
    uni1 = cor1_new[:, [0, 1]]
    idx = _unique_rows_first_indices(uni1)
    cor1_new = cor1_new[idx, :]
    cor2_new = cor2_new[idx, :]
    uni2 = cor2_new[:, [0, 1]]
    idx = _unique_rows_first_indices(uni2)
    cor1_new = cor1_new[idx, :]
    cor2_new = cor2_new[idx, :]
    parameters, rmse = lsm(cor1_new[:, :2], cor2_new[:, :2], change_form)
    solution = np.array(
        [
            [parameters[0], parameters[1], parameters[4]],
            [parameters[2], parameters[3], parameters[5]],
            [parameters[6], parameters[7], 1.0],
        ],
        dtype=float,
    )
    return solution, rmse, cor1_new, cor2_new


def fsc_with_matlab_stream(cor1, cor2, change_form, error_t, rand_stream):
    rng = _RandStream(rand_stream)
    return fsc(cor1, cor2, change_form, error_t, rng=rng)


def estimate_fsc_3x3(points1: np.ndarray, points2: np.ndarray, change_form: str, error_t: float):
    from .transform_ransac import TransformEstimation

    p1 = np.asarray(points1, dtype=np.float32).reshape(-1, 2)
    p2 = np.asarray(points2, dtype=np.float32).reshape(-1, 2)
    if p1.shape != p2.shape or p1.ndim != 2 or p1.shape[1] != 2:
        raise ValueError("points shape invalid")

    finite = np.isfinite(p1).all(axis=1) & np.isfinite(p2).all(axis=1)
    p1 = p1[finite]
    p2 = p2[finite]

    if change_form == "affine" and p1.shape[0] < 3:
        raise ValueError("not enough matches for affine")
    if change_form == "perspective" and p1.shape[0] < 4:
        raise ValueError("not enough matches for perspective")

    rng = np.random.RandomState(0)
    H, _rmse_raw, cor1_new, _cor2_new = fsc(p1, p2, change_form, float(error_t), rng=rng)
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)

    inlier_mask = np.zeros((p1.shape[0],), dtype=bool)
    if cor1_new is not None and cor1_new.size > 0:
        key_to_indices: dict[tuple[float, float], list[int]] = {}
        for i, (x, y) in enumerate(p1.tolist()):
            key_to_indices.setdefault((float(x), float(y)), []).append(i)
        for x, y in np.asarray(cor1_new, dtype=np.float32)[:, 0:2].tolist():
            key = (float(x), float(y))
            bucket = key_to_indices.get(key)
            if bucket:
                inlier_mask[bucket.pop(0)] = True

    p1h = np.concatenate([p1.astype(np.float64), np.ones((p1.shape[0], 1), dtype=np.float64)], axis=1)
    pred = (H @ p1h.T).T
    pred = pred[:, 0:2] / pred[:, 2:3]
    err = np.linalg.norm(pred - p2.astype(np.float64), axis=1)
    if inlier_mask.any():
        rmse = float(np.sqrt(np.mean(np.square(err[inlier_mask]))))
    else:
        rmse = float(np.sqrt(np.mean(np.square(err))))

    return TransformEstimation(model=f"fsc-{change_form}", H_3x3=H, inlier_mask=inlier_mask, rmse=rmse)
