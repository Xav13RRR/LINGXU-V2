import pytest

from lingxu.domain.wake_phrase import (
    DEFAULT_WAKE_PHRASE,
    normalise_wake_phrase,
)


def test_wake_phrase_normalisation_applies_nfkc_to_compatible_han() -> None:
    assert normalise_wake_phrase("呼叫\uf900中心") == "呼叫\u8c48中心"
    assert normalise_wake_phrase(DEFAULT_WAKE_PHRASE) == DEFAULT_WAKE_PHRASE


@pytest.mark.parametrize("phrase", ["指挥中", "汉" * 17])
def test_wake_phrase_requires_four_to_sixteen_han_characters(phrase: str) -> None:
    with pytest.raises(ValueError, match="4 到 16"):
        normalise_wake_phrase(phrase)


@pytest.mark.parametrize(
    "phrase",
    [
        "呼叫 AI 中心",
        "呼叫\uff0c指挥中心",
        "呼叫 指挥中心",
        "呼叫\u200b指挥中心",
        "呼叫\n指挥中心",
        "呼叫\ud800指挥中心",
    ],
)
def test_wake_phrase_rejects_non_han_input(phrase: str) -> None:
    with pytest.raises(ValueError, match="只能包含连续汉字"):
        normalise_wake_phrase(phrase)
