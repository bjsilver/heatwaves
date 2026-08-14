import numpy as np

import mwa


def _run_kernel(max_neighbour_cc, center_cc_loss, fraction=0.90):
    cc = np.array(
        [[[45.0, 47.0, 49.0], [48.0, 50.0, max_neighbour_cc], [46.0, 44.0, 43.0]]],
        dtype=np.float32,
    )
    cc_loss = np.zeros_like(cc)
    cc_loss[0, 1, 1] = center_cc_loss
    return mwa._mwa_kernel(cc, cc_loss, window_radius=1, loss_explained_fraction=fraction)


def test_mwa_kernel_selects_highest_positive_canopy_cover_neighbour():
    result = _run_kernel(max_neighbour_cc=80.0, center_cc_loss=27.0)

    np.testing.assert_array_equal(result[:, 0, 0, 0], np.array([0, 1], dtype=np.int8))


def test_mwa_kernel_rejects_center_without_higher_neighbour():
    result = _run_kernel(max_neighbour_cc=49.0, center_cc_loss=100.0)

    np.testing.assert_array_equal(
        result[:, 0, 0, 0],
        np.array([mwa.NEIGHBOUR_MAP_FILL_VALUE, mwa.NEIGHBOUR_MAP_FILL_VALUE], dtype=np.int8),
    )


def test_mwa_kernel_requires_loss_to_explain_selected_difference():
    passed = _run_kernel(max_neighbour_cc=80.0, center_cc_loss=27.0)
    failed = _run_kernel(max_neighbour_cc=80.0, center_cc_loss=26.9)

    np.testing.assert_array_equal(passed[:, 0, 0, 0], np.array([0, 1], dtype=np.int8))
    np.testing.assert_array_equal(
        failed[:, 0, 0, 0],
        np.array([mwa.NEIGHBOUR_MAP_FILL_VALUE, mwa.NEIGHBOUR_MAP_FILL_VALUE], dtype=np.int8),
    )


def test_mwa_kernel_accepts_custom_loss_explained_fraction():
    result = _run_kernel(max_neighbour_cc=80.0, center_cc_loss=15.0, fraction=0.50)

    np.testing.assert_array_equal(result[:, 0, 0, 0], np.array([0, 1], dtype=np.int8))