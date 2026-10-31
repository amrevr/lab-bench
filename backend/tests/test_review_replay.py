from labbench.pipeline.sam import Track
from labbench.replay import persist_tracks


def _tr(i, label, y):
    return Track(id=i, label=label, boxes=[(t, 0.1, y, 0.3, y + 0.2) for t in (0.0, 1.0)])


def test_ids_persist_and_never_collide_across_chunks():
    counts, prev, seen = {}, [], []
    # same hand under raw ids hand:0, hand:0, hand:3; chunk 3 also has an unmatched new hand with raw id hand:0
    chunks = [[_tr("hand:0", "hand", 0.1)], [_tr("hand:0", "hand", 0.1)],
              [_tr("hand:3", "hand", 0.1), _tr("hand:0", "hand", 0.7)]]
    for tracks in chunks:
        link = persist_tracks(prev, tracks, counts)
        seen.append(link)
        prev = tracks or prev
    assert seen[0]["hand:0"] == seen[1]["hand:0"] == seen[2]["hand:3"]
    assert seen[2]["hand:0"] not in (seen[0]["hand:0"], seen[2]["hand:3"])
