import numpy as np
from fsc import fsc, fsc_with_matlab_stream


def _compute_iterations(M, change_form):
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
    return n, iterations


def main():
    matches_path = r"d:\hand_craft_registration\WSSF-main\WSSF-main\save_image_compare\matches_wssf.txt"
    data = np.loadtxt(matches_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    max_rows = min(200, data.shape[0])
    data = data[:max_rows, :]
    cor1 = data[:, :2]
    cor2 = data[:, 2:4]
    change_form = "affine"
    error_t = 3.0
    n, iterations = _compute_iterations(cor1.shape[0], change_form)
    try:
        import matlab.engine

        eng = matlab.engine.start_matlab()
        eng.addpath(r"d:\hand_craft_registration\WSSF-main\WSSF-main\Others", nargout=0)
        seed = 0.0
        eng.rng(seed, "twister", nargout=0)
        stream_len = int(iterations * n * 50)
        rand_stream = np.array(eng.rand(1, stream_len)).reshape(-1)
        eng.rng(seed, "twister", nargout=0)
        cor1_mat = matlab.double(cor1.tolist())
        cor2_mat = matlab.double(cor2.tolist())
        solution_m, rmse_m, cor1_new_m, cor2_new_m = eng.FSC(
            cor1_mat, cor2_mat, change_form, error_t, nargout=4
        )
        eng.quit()
        solution_m = np.array(solution_m, dtype=float)
        rmse_m = float(rmse_m)
        cor1_new_m = np.array(cor1_new_m, dtype=float)
        cor2_new_m = np.array(cor2_new_m, dtype=float)
        solution_p, rmse_p, cor1_new_p, cor2_new_p = fsc_with_matlab_stream(
            cor1, cor2, change_form, error_t, rand_stream
        )
        solution_diff = np.max(np.abs(solution_m - solution_p))
        rmse_diff = abs(rmse_m - rmse_p)
        cor1_diff = np.max(np.abs(cor1_new_m - cor1_new_p))
        cor2_diff = np.max(np.abs(cor2_new_m - cor2_new_p))
        print("solution_max_abs_diff", solution_diff)
        print("rmse_abs_diff", rmse_diff)
        print("cor1_new_max_abs_diff", cor1_diff)
        print("cor2_new_max_abs_diff", cor2_diff)
    except Exception as exc:
        solution_p, rmse_p, cor1_new_p, cor2_new_p = fsc(
            cor1, cor2, change_form, error_t, rng=np.random.RandomState(0)
        )
        print("matlab_engine_unavailable", str(exc))
        print("python_solution", solution_p)
        print("python_rmse", rmse_p)
        print("python_cor1_new_shape", cor1_new_p.shape)
        print("python_cor2_new_shape", cor2_new_p.shape)


if __name__ == "__main__":
    main()
