"""Leave-one-out detectability: a step the aligner marks done in < THETA of the OTHER clips (same recipe, real events,
no labels) is set visible=False for this clip, so its absence raises no skip. Writes realigned sessions to OUT.
Usage: detectability.py RECIPE SESSIONS OUT THETA"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))
from labbench.eval import realign
from labbench.pipeline.align import Aligner
from labbench.schema import Protocol, Event
R, S, O, TH = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), float(sys.argv[4])
proto = Protocol.model_validate_json((pathlib.Path(__file__).resolve().parents[1] / f"fixtures/recipe{R}/protocol.json").read_text())
sess = {p.name.split(".")[0]: json.loads(p.read_text()) for p in sorted(S.glob("*.session.json")) if p.name.split(".")[0] != proto.source_clip}
def done(s):
    al = Aligner(proto)
    for e in s["events"]: al.feed(Event.model_validate(e))
    return set(al.done)
D = {c: done(s) for c, s in sess.items()}; O.mkdir(parents=True, exist_ok=True); hidden = {}
for c, s in sess.items():
    others = [D[o] for o in D if o != c]
    rate = {st.id: sum(st.id in d for d in others) / len(others) for st in proto.steps}
    p = proto.model_copy(deep=True)
    for st in p.steps:
        if rate[st.id] < TH: st.visible = False
    hidden[c] = sorted(st.id for st in p.steps if not st.visible)
    (O / f"{c}.session.json").write_text(json.dumps(realign(s, p)))
print("hidden per clip (sample):", dict(list(hidden.items())[:3]))
