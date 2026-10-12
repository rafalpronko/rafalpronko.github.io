import math
import random

import pytest

from cosine_lab import (
    Recipe,
    check_same_recipe,
    cosine_similarity,
    mean_pool,
    norm,
    normalize,
)


def test_norm_is_euclidean_length():
    assert norm([3.0, 4.0]) == 5.0
    assert norm([1.0, 2.0, 2.0]) == 3.0


def test_identical_direction_is_one():
    a = [1.0, 2.0, 3.0]
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_opposite_direction_is_minus_one():
    assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_orthogonal_is_zero():
    assert cosine_similarity([1.0, 0.0, 0.0], [0.0, 5.0, 0.0]) == 0.0


def test_positive_scaling_does_not_change_cosine():
    a, b = [0.2, -0.7, 1.5], [1.0, 0.3, 0.4]
    scaled = [1000.0 * x for x in a]
    assert cosine_similarity(scaled, b) == pytest.approx(
        cosine_similarity(a, b)
    )


def test_negative_scaling_flips_sign():
    a, b = [0.2, -0.7, 1.5], [1.0, 0.3, 0.4]
    flipped = [-3.0 * x for x in a]
    assert cosine_similarity(flipped, b) == pytest.approx(
        -cosine_similarity(a, b)
    )


def test_symmetric():
    a, b = [0.5, 0.1, -0.2], [0.3, 0.9, 0.0]
    assert cosine_similarity(a, b) == cosine_similarity(b, a)


def test_empty_vectors_rejected():
    with pytest.raises(ValueError, match="empty"):
        cosine_similarity([], [])


def test_mismatched_dimensions_rejected():
    with pytest.raises(ValueError, match="dimension"):
        cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0])


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        cosine_similarity([1.0, bad], [1.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        cosine_similarity([1.0, 1.0], [bad, 1.0])


def test_zero_vector_rejected_instead_of_returning_a_score():
    with pytest.raises(ValueError, match="zero"):
        cosine_similarity([0.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="zero"):
        normalize([0.0, 0.0, 0.0])


def test_large_magnitudes_do_not_overflow():
    # A naive dot product of these vectors would overflow to inf.
    a = [1e200, 1e200]
    b = [2e200, 2e200]
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_supports_vectors_whose_raw_norm_overflows():
    # The length cannot be represented, but the direction can.
    a = [1.7e308] * 4
    assert math.isinf(norm(a))
    assert cosine_similarity(a, [1.0] * 4) == pytest.approx(1.0)


def test_subnormal_cosine_preserves_direction():
    tiny = 5e-324
    score = cosine_similarity([tiny, tiny], [1.0, 0.0])
    assert score == pytest.approx(math.sqrt(0.5))


@pytest.mark.parametrize("exponent", [1, 52, 1074])
def test_subnormal_positive_power_two_scaling_keeps_cosine(exponent):
    tiny = 5e-324
    larger = math.ldexp(tiny, exponent)
    original = cosine_similarity([tiny, tiny], [1.0, 0.0])
    rescaled = cosine_similarity([larger, larger], [1.0, 0.0])
    assert original == pytest.approx(rescaled)


def test_normalize_subnormal_vector_gives_unit_length():
    unit = normalize([5e-324, 5e-324])
    assert unit == pytest.approx((math.sqrt(0.5), math.sqrt(0.5)))
    assert norm(unit) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "vector, message",
    [([], "empty"), ([math.nan], "finite"),
     ([math.inf], "finite"), ([-math.inf], "finite")],
)
def test_norm_validates_input(vector, message):
    with pytest.raises(ValueError, match=message):
        norm(vector)


def test_tiny_magnitudes_do_not_underflow_to_zero_norm():
    # A naive sum of squares would underflow to 0.0 for these values.
    a = [1e-200, 0.0]
    b = [3e-200, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_result_stays_in_closed_interval():
    rng = random.Random(20231214)
    for _ in range(2000):
        dim = rng.randint(1, 16)
        a = [rng.uniform(-1, 1) for _ in range(dim)]
        b = [rng.uniform(-1, 1) for _ in range(dim)]
        if norm(a) == 0.0 or norm(b) == 0.0:
            continue
        assert -1.0 <= cosine_similarity(a, b) <= 1.0
        assert math.isclose(cosine_similarity(a, a), 1.0, rel_tol=1e-12)


def test_normalize_gives_unit_length():
    assert norm(normalize([3.0, 4.0])) == pytest.approx(1.0)
    assert normalize([3.0, 4.0]) == pytest.approx((0.6, 0.8))


def test_inner_product_of_normalized_vectors_equals_cosine():
    a, b = [0.2, -0.7, 1.5], [1.0, 0.3, 0.4]
    ua, ub = normalize(a), normalize(b)
    dot = math.fsum(x * y for x, y in zip(ua, ub))
    assert dot == pytest.approx(cosine_similarity(a, b))


def test_squared_distance_of_unit_vectors_is_two_minus_two_cosine():
    a, b = [0.2, -0.7, 1.5], [1.0, 0.3, 0.4]
    ua, ub = normalize(a), normalize(b)
    squared_distance = math.fsum((x - y) ** 2 for x, y in zip(ua, ub))
    assert squared_distance == pytest.approx(
        2.0 - 2.0 * cosine_similarity(a, b)
    )


def test_mean_pool_averages_per_dimension():
    assert mean_pool([[1.0, 0.0], [3.0, 2.0]]) == pytest.approx((2.0, 1.0))


def test_mean_pool_rejects_empty_and_ragged_input():
    with pytest.raises(ValueError, match="empty"):
        mean_pool([])
    with pytest.raises(ValueError, match="dimension"):
        mean_pool([[1.0, 2.0], [1.0]])


@pytest.mark.parametrize("rows", [[[]], [[], []]])
def test_mean_pool_rejects_empty_rows(rows):
    with pytest.raises(ValueError, match="empty"):
        mean_pool(rows)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_mean_pool_validates_every_row(bad):
    for rows in ([[bad, 0.0], [1.0, 0.0]],
                 [[1.0, 0.0], [bad, 0.0]]):
        with pytest.raises(ValueError, match="finite"):
            mean_pool(rows)


@pytest.mark.parametrize(
    "rows, expected",
    [([[1e308], [1e308]], (1e308,)),
     ([[1.7e308], [1.7e308]], (1.7e308,)),
     ([[1e308], [-1e308]], (0.0,))],
)
def test_mean_pool_avoids_raw_sum_overflow(rows, expected):
    assert mean_pool(rows) == pytest.approx(expected)


def test_mean_pool_handles_zero_and_subnormal_columns():
    assert mean_pool([[0.0, 5e-324], [0.0, 5e-324]]) == (0.0, 5e-324)


def test_same_recipe_passes():
    r = Recipe(
        encoder="toy-encoder", version="1", pooling="mean", normalized=True
    )
    check_same_recipe(r, r)


@pytest.mark.parametrize(
    "query",
    [
        Recipe("other-encoder", "1", "mean", True),
        Recipe("toy-encoder", "2", "mean", True),
        Recipe("toy-encoder", "1", "cls", True),
        Recipe("toy-encoder", "1", "mean", False),
    ],
)
def test_any_recipe_difference_is_rejected(query):
    index = Recipe(
        encoder="toy-encoder", version="1", pooling="mean", normalized=True
    )
    with pytest.raises(ValueError, match="rebuild"):
        check_same_recipe(index, query)
