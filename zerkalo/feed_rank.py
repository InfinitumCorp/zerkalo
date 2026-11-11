"""Rank a draft the way X ranks For You.

This is a Python translation of the value model in
https://github.com/xai-org/x-algorithm (Apache-2.0), files
`xai-value-model/scoring.rs` and `home-mixer/params/param.rs`.
The weights multiply a predicted probability or a predicted
duration. They do not multiply raw like or repost counts.

Not included, on purpose: the Phoenix model, visibility filters,
takedowns, and the moderation rules. Those are not a marketing score.

`FOR_YOU_WEIGHTS` are the numbers published in that repository on
2026-11-11. A campaign can pass its own weights.
"""

from __future__ import annotations

NEGATIVE_SCORES_OFFSET = 0.001

# Published For You weights. Negative actions are stored as negative numbers.
FOR_YOU_WEIGHTS = {
    "favorite": 0.708,
    "reply": 5.0,
    "retweet": 1.0,
    "photo_expand": 0.05,
    "video_open": 0.0,
    "click": 0.274,
    "open_link": 0.2,
    "profile_click": 0.0,
    "vqv": 0.0,
    "share": 2.118,
    "share_via_dm": 5.0,
    "share_via_copy_link": 20.0,
    "dwell": 0.05,
    "quote": 5.0,
    "quoted_click": 0.05,
    "quoted_vqv": 0.0,
    "follow_author": 4.0,
    "post_unexplored": 0.015,
    "post_video": 0.0,
    "grok_analyze": 0.0,
    "dwell_time": 0.0075,
    "click_dwell_time": 0.385,
    "video_continuation": 0.0,
    "user_video_continuation": 0.0,
    "profile_visit_secs": 0.0,
    "not_interested": -53.07,
    "block_author": -31.2,
    "mute_author": -58.8,
    "report": -234.0,
    "not_dwelled": -0.02,
}

POSITIVE = (
    "favorite", "reply", "retweet", "photo_expand", "video_open", "click",
    "open_link", "profile_click", "vqv", "share", "share_via_dm",
    "share_via_copy_link", "dwell", "quote", "quoted_click", "quoted_vqv",
    "follow_author", "post_unexplored", "post_video", "grok_analyze",
)
NEGATIVE = ("not_interested", "block_author", "mute_author", "report", "not_dwelled")


def _w(weights, name):
    return float(weights.get(name, 0.0) or 0.0)


def _p(pred, name):
    if not pred or pred.get(name) is None:
        return 0.0
    return float(pred[name])


def positive_sum(weights):
    return sum(_w(weights, name) for name in POSITIVE)


def negative_sum(weights):
    # The stored weights are negative. This returns their magnitude.
    return -sum(_w(weights, name) for name in NEGATIVE)


def total_sum(weights):
    return positive_sum(weights) + negative_sum(weights)


def offset_score(combined, weights):
    total = total_sum(weights)
    if total == 0.0:
        return max(combined, 0.0)
    if combined < 0.0:
        return (combined + negative_sum(weights)) / total * NEGATIVE_SCORES_OFFSET
    return combined + NEGATIVE_SCORES_OFFSET


def weighted_terms(weights, pred, *, in_network=True, mutual=False, vqv_eligible=False):
    """Return (name, contribution) in the published head order."""
    reply_w = _w(weights, "reply")
    dwell_w = _w(weights, "dwell")
    if mutual:
        reply_w += _w(weights, "reply_boost")
        dwell_w += _w(weights, "dwell_boost")
    vqv_w = _w(weights, "vqv") if vqv_eligible else 0.0
    quoted_vqv_w = _w(weights, "quoted_vqv") if pred and pred.get("quoted_vqv_eligible") else 0.0
    include_unexplored = bool(weights.get("unexplored_out_of_network")) or in_network is True
    unexplored_w = _w(weights, "post_unexplored") if include_unexplored else 0.0
    video_cont = _p(pred, "video_open") * _p(pred, "video_continuation_secs")
    profile_secs = _p(pred, "profile_click") * _p(pred, "profile_visit_secs")
    rows = [
        ("favorite", _p(pred, "favorite") * _w(weights, "favorite")),
        ("reply", _p(pred, "reply") * reply_w),
        ("retweet", _p(pred, "retweet") * _w(weights, "retweet")),
        ("photo_expand", _p(pred, "photo_expand") * _w(weights, "photo_expand")),
        ("video_open", _p(pred, "video_open") * _w(weights, "video_open")),
        ("click", _p(pred, "click") * _w(weights, "click")),
        ("open_link", _p(pred, "open_link") * _w(weights, "open_link")),
        ("profile_click", _p(pred, "profile_click") * _w(weights, "profile_click")),
        ("vqv", _p(pred, "vqv") * vqv_w),
        ("share", _p(pred, "share") * _w(weights, "share")),
        ("share_via_dm", _p(pred, "share_via_dm") * _w(weights, "share_via_dm")),
        ("share_via_copy_link", _p(pred, "share_via_copy_link") * _w(weights, "share_via_copy_link")),
        ("dwell", _p(pred, "dwell") * dwell_w),
        ("quote", _p(pred, "quote") * _w(weights, "quote")),
        ("quoted_click", _p(pred, "quoted_click") * _w(weights, "quoted_click")),
        ("quoted_vqv", _p(pred, "quoted_vqv") * quoted_vqv_w),
        ("dwell_time", _p(pred, "dwell_time") * _w(weights, "dwell_time")),
        ("click_dwell_time", _p(pred, "click_dwell_time") * _w(weights, "click_dwell_time")),
        ("video_continuation", video_cont * _w(weights, "video_continuation")),
        ("profile_visit_secs", profile_secs * _w(weights, "profile_visit_secs")),
        ("follow_author", _p(pred, "follow_author") * _w(weights, "follow_author")),
        ("not_interested", _p(pred, "not_interested") * _w(weights, "not_interested")),
        ("block_author", _p(pred, "block_author") * _w(weights, "block_author")),
        ("mute_author", _p(pred, "mute_author") * _w(weights, "mute_author")),
        ("report", _p(pred, "report") * _w(weights, "report")),
        ("not_dwelled", _p(pred, "not_dwelled") * _w(weights, "not_dwelled")),
        ("post_unexplored", _p(pred, "post_unexplored") * unexplored_w),
        ("post_video", _p(pred, "post_video") * _w(weights, "post_video")),
        ("grok_analyze", _p(pred, "grok_analyze") * _w(weights, "grok_analyze")),
    ]
    return rows


def combine(weights, pred, **flags):
    return sum(value for _, value in weighted_terms(weights, pred, **flags))


def diversity_multiplier(decay, floor, seen):
    return (1.0 - floor) * (decay ** seen) + floor


def score_candidates(candidates, weights=None, *, oon_weight=1.0, rescore_in_network_replies=False,
                     author_diversity=False, diversity_decay=0.5, diversity_floor=0.25):
    """Score and order drafts.

    Each candidate is a dict: predictions (action name -> probability or
    seconds), author, in_network, mutual, reply, vqv_eligible.
    Returns the same dicts with `weighted` and `score` set, highest first.
    The existing rehearsal feed is not changed by this.
    """
    weights = FOR_YOU_WEIGHTS if weights is None else weights
    prepared = []
    for candidate in candidates:
        pred = candidate.get("predictions") or {}
        net = combine(
            weights,
            pred,
            in_network=candidate.get("in_network", True),
            mutual=bool(candidate.get("mutual")),
            vqv_eligible=bool(candidate.get("vqv_eligible")),
        )
        weighted = offset_score(net, weights)
        prepared.append((candidate, weighted))

    order = sorted(range(len(prepared)), key=lambda i: prepared[i][1], reverse=True)
    seen = {}
    ranked = []
    for index in order:
        candidate, weighted = prepared[index]
        author = candidate.get("author")
        k = seen.get(author, 0)
        seen[author] = k + 1
        multiplier = 1.0
        if author_diversity:
            multiplier *= diversity_multiplier(diversity_decay, diversity_floor, k)
        out_of_network = candidate.get("in_network") is False
        reply_or_repost = bool(candidate.get("reply") or candidate.get("repost"))
        if oon_weight != 1.0 and (out_of_network or (rescore_in_network_replies and reply_or_repost)):
            multiplier *= oon_weight
        scored = dict(candidate)
        scored["weighted"] = weighted
        scored["score"] = weighted * multiplier
        ranked.append(scored)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked
