import pytest

from low_latency_llm_serving.inference import GenerationSettings


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (GenerationSettings(max_new_tokens=0), "max_new_tokens"),
        (GenerationSettings(temperature=0), "temperature"),
        (GenerationSettings(top_p=1.1), "top_p"),
    ],
)
def test_invalid_generation_settings(
    settings: GenerationSettings, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        settings.validate()


def test_default_generation_settings_are_valid() -> None:
    GenerationSettings().validate()

