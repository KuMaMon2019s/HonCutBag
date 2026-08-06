import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from quality.delivery_promise import DeliveryPromise, PromiseType, classify_from_brief

def test_all_eight_promise_types_have_rules():
    assert len(PromiseType) == 8
    assert all(DeliveryPromise(kind, False, False, "corporate", "presentable").get_rules() for kind in PromiseType)

def test_round_trip_serialization():
    original = classify_from_brief("cinematic", {"quality": "broadcast"})
    assert DeliveryPromise.from_dict(original.to_dict()) == original

def test_motion_led_rejects_slide_fallback():
    promise = classify_from_brief("cinematic", {})
    report = promise.validate_cuts([{"type": "text_card"}, {"source": "clip.mp4"}, {"source": "still.png"}])
    assert not report["valid"] and report["motion_ratio"] == 1 / 3
