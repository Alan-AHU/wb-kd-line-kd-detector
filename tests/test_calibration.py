import numpy as np

from src.kd_detector import fit_log_kd_mapping, make_piecewise_kd_mapper, y_to_kd


def test_log_kd_mapping_close_to_reference_points() -> None:
    y = [322, 360, 398, 452, 487, 512, 525]
    refs = np.array([250.0, 150.0, 100.0, 70.0, 50.0, 40.0, 35.0], dtype=float)

    a, b = fit_log_kd_mapping(y, refs)
    predicted = np.array([y_to_kd(v, a, b) for v in y])

    relative_error = np.abs(predicted - refs) / refs
    assert float(relative_error.max()) < 0.16


def test_interpolated_70kd_location_is_reasonable() -> None:
    y = [322, 360, 398, 452, 487, 512, 525]
    a, b = fit_log_kd_mapping(y)

    kd_around_70 = y_to_kd(452, a, b)
    assert 55.0 <= kd_around_70 <= 85.0


def test_piecewise_mapper_hits_scale_points_precisely() -> None:
    y = [322, 360, 398, 452, 487, 512, 525]
    refs = np.array([250.0, 150.0, 100.0, 70.0, 50.0, 40.0, 35.0], dtype=float)
    mapper = make_piecewise_kd_mapper(y, refs)

    pred = np.array([mapper(v) for v in y], dtype=float)
    rel_err = np.abs(pred - refs) / refs
    assert float(rel_err.max()) < 0.02
