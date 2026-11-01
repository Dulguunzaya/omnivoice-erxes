"""erxes narration should retain English UI pronunciations in Mongolian."""

from types import SimpleNamespace

import numpy as np
import pytest

from omnivoice import OmniVoiceGenerationConfig
from omnivoice.cli.demo import build_demo
from omnivoice.models.omnivoice import GenerationTask, OmniVoice
from omnivoice.utils import text as text_utils


def test_mongolian_narration_preserves_english_callouts():
    from omnivoice.utils.erxes import ERXES_PRONUNCIATIONS

    result = text_utils.apply_pronunciation_overrides(
        "Өнөөдөр SidebarPanel-ийн Properties болон Widgetийг шинэчиллээ.",
        ERXES_PRONUNCIATIONS,
    )
    assert result == (
        "Өнөөдөр [S AY1 D B AA2 R P AE1 N AH0 L]-ийн "
        "[P R AA1 P ER0 T IY0 Z] болон [W IH1 JH AH0 T]ийг шинэчиллээ."
    )


@pytest.mark.parametrize(
    "term",
    [
        "sidebarpanel",
        "SIDEBAR PANEL",
        "sidebar-panel",
        "sidebar_panel",
        "Sidebar  Panel",
    ],
)
def test_component_name_spelling_variants(term):
    from omnivoice.utils.erxes import ERXES_PRONUNCIATIONS

    assert text_utils.apply_pronunciation_overrides(term, ERXES_PRONUNCIATIONS) == (
        "[S AY1 D B AA2 R P AE1 N AH0 L]"
    )


def test_longest_names_win_without_changing_unrelated_words_or_inline_controls():
    result = text_utils.apply_pronunciation_overrides(
        "sidebar panel, sidebars, my_sidebar, [sidebar], [laughter], [B EY1 S], Монгол.",
        {
            "sidebar": "S AY1 D B AA2 R",
            "sidebar panel": "S AY1 D B AA2 R P AE1 N AH0 L",
        },
    )
    assert result == (
        "[S AY1 D B AA2 R P AE1 N AH0 L], sidebars, my_sidebar, "
        "[sidebar], [laughter], [B EY1 S], Монгол."
    )


@pytest.mark.parametrize("overrides", [None, {}])
def test_disabled_overrides_leave_text_unchanged(overrides):
    text = "  Properties болон Widgetийг шинэчиллээ.  "
    assert text_utils.apply_pronunciation_overrides(text, overrides) == text


def test_overrides_reach_encoding_after_duration_estimation_without_touching_reference():
    """Catch applying long phoneme strings before timing/chunking or to clone transcripts."""
    config = OmniVoiceGenerationConfig(
        pronunciation_overrides={"widget": "W IH1 JH AH0 T"}
    )
    task = GenerationTask(
        batch_size=1,
        texts=["Widgetийг шинэчиллээ."],
        target_lens=[25],
        langs=["mn"],
        instructs=["male"],
        ref_texts=["Widget reference."],
        ref_audio_tokens=[None],
        ref_rms=[None],
    )

    class ReachedEncoding(Exception):
        pass

    def encode(text, duration, ref_text, ref_tokens, lang, instruct, denoise):
        assert text == "[W IH1 JH AH0 T]ийг шинэчиллээ."
        assert duration == 25
        assert ref_text == "Widget reference."
        assert lang == "mn"
        assert instruct == "male"
        raise ReachedEncoding

    with pytest.raises(ReachedEncoding):
        OmniVoice._generate_iterative(
            SimpleNamespace(_prepare_inference_inputs=encode), task, config
        )
    assert task.texts == ["Widgetийг шинэчиллээ."]


class DemoModel:
    sampling_rate = 24000

    def __init__(self):
        self.calls = []
        self.clone_calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [np.zeros(8, dtype=np.float32)]

    def create_voice_clone_prompt(self, **kwargs):
        self.clone_calls.append(kwargs)
        return "clone-prompt"


@pytest.fixture(scope="module")
def demo():
    model = DemoModel()
    blocks = build_demo(model, "test-checkpoint")
    yield blocks, model
    blocks.close()


@pytest.mark.parametrize("mode", ["design", "clone"])
@pytest.mark.parametrize("enabled", [False, True])
def test_erxes_toggle_controls_both_generation_paths(demo, mode, enabled):
    blocks, model = demo
    handler = next(fn for fn in blocks.fns.values() if fn.name == f"_{mode}_fn")
    dependency = next(
        d for d in blocks.config["dependencies"] if d["id"] == handler._id
    )
    components = {c["id"]: c for c in blocks.config["components"]}
    values = []
    erxes_switches = []
    for component_id in dependency["inputs"]:
        component = components[component_id]
        props = component["props"]
        label = props.get("label", "")
        value = props.get("value")
        if label.startswith("Text to Synthesize"):
            value = "Properties-ийг шинэчиллээ."
        elif label.startswith("Language"):
            value = "Mongolian"
        elif label == "Reference Audio / 参考音频":
            value = "reference.wav"
        elif label.startswith("Reference Text"):
            value = "Properties reference."
        elif label.startswith("erxes"):
            erxes_switches.append(component)
            assert value is False
            value = enabled
        values.append(value)

    assert len(erxes_switches) == 1
    audio, status = handler.fn(*values)
    assert audio[0] == 24000
    assert status.startswith("Done.")
    call = model.calls[-1]
    assert call["text"] == "Properties-ийг шинэчиллээ."
    assert call["language"] == "Mongolian"
    overrides = call["generation_config"].pronunciation_overrides
    if enabled:
        assert text_utils.apply_pronunciation_overrides(call["text"], overrides) == (
            "[P R AA1 P ER0 T IY0 Z]-ийг шинэчиллээ."
        )
    else:
        assert overrides is None
    if mode == "clone":
        assert model.clone_calls[-1] == {
            "ref_audio": "reference.wav",
            "ref_text": "Properties reference.",
        }


@pytest.mark.parametrize("mode", ["design", "clone"])
@pytest.mark.parametrize("enabled", [False, True])
def test_existing_custom_wrappers_work_or_explain_unsupported_erxes_mode(mode, enabled):
    calls = []

    def wrapper(
        text,
        language,
        ref_audio,
        instruct,
        ns,
        gs,
        dn,
        sp,
        du,
        pp,
        po,
        mode,
        ref_text=None,
    ):
        calls.append((text, language, mode, ref_text))
        return None, "Wrapped"

    blocks = build_demo(DemoModel(), "test-checkpoint", generate_fn=wrapper)
    try:
        handler = next(fn for fn in blocks.fns.values() if fn.name == f"_{mode}_fn")
        if mode == "design":
            result = handler.fn(
                "Widgetийг шинэчиллээ.",
                "Mongolian",
                32,
                2,
                True,
                1,
                None,
                True,
                True,
                enabled,
                *("Auto",) * 6,
            )
        else:
            result = handler.fn(
                "Widgetийг шинэчиллээ.",
                "Mongolian",
                "ref.wav",
                "Reference.",
                "",
                32,
                2,
                True,
                1,
                None,
                True,
                True,
                enabled,
            )
        if enabled:
            assert result[0] is None
            assert "custom generation wrapper" in result[1]
            assert calls == []
            return
        assert result == (None, "Wrapped")
        assert calls == [
            (
                "Widgetийг шинэчиллээ.",
                "Mongolian",
                mode,
                "Reference." if mode == "clone" else None,
            )
        ]
    finally:
        blocks.close()
