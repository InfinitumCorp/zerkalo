"""Parity with the hand-computed test in xai-org/x-algorithm scoring.rs."""

from zerkalo.feed_rank import FOR_YOU_WEIGHTS, score_candidates

WEIGHTS = {
    "favorite": 2.0,
    "reply": 4.0,
    "retweet": 1.0,
    "dwell": 0.5,
    "vqv": 3.0,
    "dwell_time": 0.1,
    "click_dwell_time": 0.2,
    "post_unexplored": 1.5,
    "post_video": 2.5,
    "report": -100.0,
    "not_interested": -10.0,
    "reply_boost": 1.0,
    "dwell_boost": 0.5,
}


def test_hand_computed_parity():
    offset = 0.001
    mutual = 2.0 * 0.5 + 5.0 * 0.1 + 1.0 * 0.4 + 0.1 * 10.0 + 1.5 * 0.2 + 2.5 * 0.04 + offset
    out_of_network = 2.0 * 0.2 + 0.2 * 5.0 + offset
    same_author = 2.0 * 0.05 + 3.0 * 0.5 + 0.2 * 5.0 + offset
    net_report = 2.0 * 0.01 - 100.0 * 0.05
    reported = (net_report + 110.0) / (14.5 + 110.0) * offset
    second = (1.0 - 0.25) * 0.5 + 0.25

    ranked = score_candidates(
        [
            {"id": "mutual", "author": 1, "in_network": True, "mutual": True,
             "predictions": {"favorite": 0.5, "reply": 0.1, "dwell": 0.4, "dwell_time": 10.0,
                             "post_unexplored": 0.2, "post_video": 0.04}},
            {"id": "oon", "author": 2, "in_network": False, "vqv_eligible": False,
             "predictions": {"favorite": 0.2, "vqv": 0.9, "click_dwell_time": 5.0, "post_unexplored": 0.2}},
            {"id": "video", "author": 1, "in_network": True, "vqv_eligible": True,
             "predictions": {"favorite": 0.05, "vqv": 0.5, "click_dwell_time": 5.0}},
            {"id": "report", "author": 3, "in_network": True, "reply": True, "mutual": True,
             "predictions": {"favorite": 0.01, "report": 0.05}},
        ],
        WEIGHTS,
        oon_weight=0.75,
        rescore_in_network_replies=True,
        author_diversity=True,
    )
    by_id = {item["id"]: item for item in ranked}
    assert abs(by_id["mutual"]["weighted"] - mutual) < 1e-12
    assert abs(by_id["oon"]["weighted"] - out_of_network) < 1e-12
    assert abs(by_id["video"]["weighted"] - same_author) < 1e-12
    assert abs(by_id["report"]["weighted"] - reported) < 1e-12
    assert abs(by_id["mutual"]["score"] - mutual) < 1e-12
    assert abs(by_id["oon"]["score"] - out_of_network * 0.75) < 1e-12
    assert abs(by_id["video"]["score"] - same_author * second) < 1e-12
    assert abs(by_id["report"]["score"] - reported * 0.75) < 1e-12
    assert [item["id"] for item in ranked] == ["mutual", "video", "oon", "report"]


def test_published_weights_penalise_a_likely_report():
    liked = score_candidates([{"author": "a", "predictions": {"favorite": 0.2, "dwell": 0.5}}])
    reported = score_candidates([{"author": "a", "predictions": {"favorite": 0.2, "dwell": 0.5, "report": 0.05}}])
    assert reported[0]["score"] < liked[0]["score"]
    assert FOR_YOU_WEIGHTS["report"] == -234.0
    assert FOR_YOU_WEIGHTS["share_via_copy_link"] == 20.0
