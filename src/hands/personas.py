import os
import random
from dataclasses import dataclass


# A BiometricPersona is one imaginary human. It has two halves:
#   MOTOR DNA    -> feeds the Hands: how the finger moves (tremor, speed, ...).
#   DECISION DNA -> feeds the Planner/loop: what the person decides (how long
#                   they keep trying, whether they wander).
# The grounding model never sees any of this -- it stays persona-blind.
#
# A persona is fully determined by its integer SEED plus its ARCHETYPE:
# same seed + same archetype -> the same human, every run. Each archetype is a
# recognizable TYPE of person (a fast power-user, a careful deliberate user, ...)
# with its own trait ranges; the seed then picks one specific human inside that
# type, so two "speedy" users still differ from each other.


@dataclass
class BiometricPersona:
    name: str
    seed: int
    archetype: str

    # --- MOTOR DNA (Hands) ---
    wpm: float
    typo_rate: float
    burstiness: float
    scatter_px: float
    fitts_precision: float
    reaction_ms: float
    dwell_scale: float
    curve_bias: float
    tremor_scale: float
    flick_vigor: float
    overshoot_rate: float
    fatigue_rate: float
    fling_bias: float   # 0.0 always careful slow scrolls .. 1.0 hard fast flings

    # --- DECISION DNA (Planner / loop) ---
    curiosity: float    # 0.0 laser-focused .. 1.0 easily distracted
    patience: float     # 0.0 gives up fast .. 1.0 keeps trying

    @property
    def retry_budget(self) -> int:
        # Impatient personas retry a stuck step only once or twice; patient ones
        # keep at it. 1..5 attempts beyond the first.
        return 1 + round(self.patience * 4)


# Default spread for any trait an archetype doesn't pin down.
_BASE = {
    "wpm": (22.0, 58.0),
    "typo_rate": (0.015, 0.14),
    "burstiness": (1.0, 2.2),
    "scatter_px": (1.8, 6.2),
    "fitts_precision": (0.65, 1.30),
    "reaction_ms": (150.0, 330.0),
    "dwell_scale": (0.72, 1.40),
    "curve_bias": (0.30, 1.00),     # magnitude only; sign = handedness
    "tremor_scale": (0.60, 1.50),
    "flick_vigor": (0.82, 1.28),
    "overshoot_rate": (0.0, 0.32),
    "fatigue_rate": (0.0, 1.0),
    "fling_bias": (0.15, 0.85),
    "curiosity": (0.1, 0.9),
    "patience": (0.1, 0.9),
}


# Each archetype overrides only the traits that DEFINE it; the rest fall back to
# _BASE. Ranges (not fixed values) so every seed is still a distinct individual.
ARCHETYPES = {
    # Fast, confident power-user: hard fast flings, quick taps, little hesitation,
    # types fast (so more typos), impatient, not very curious.
    "speedy": {
        "wpm": (46, 62), "typo_rate": (0.05, 0.14), "burstiness": (1.6, 2.4),
        "reaction_ms": (140, 210), "dwell_scale": (0.65, 0.90),
        "fitts_precision": (0.85, 1.15), "tremor_scale": (0.55, 0.95),
        "flick_vigor": (1.12, 1.35), "overshoot_rate": (0.10, 0.30),
        "fling_bias": (0.70, 0.95), "patience": (0.15, 0.45), "curiosity": (0.15, 0.50),
    },
    # Slow, deliberate: short careful scrolls, precise taps, patient, few typos.
    "careful": {
        "wpm": (24, 40), "typo_rate": (0.01, 0.05), "burstiness": (1.0, 1.4),
        "reaction_ms": (240, 340), "dwell_scale": (1.05, 1.45),
        "fitts_precision": (1.05, 1.35), "tremor_scale": (0.55, 0.95),
        "flick_vigor": (0.80, 1.02), "overshoot_rate": (0.0, 0.10),
        "fling_bias": (0.10, 0.35), "patience": (0.60, 0.95), "curiosity": (0.20, 0.60),
    },
    # Balanced, typical user: middle of every trait.
    "average": {
        "wpm": (32, 48), "typo_rate": (0.03, 0.09), "burstiness": (1.3, 1.9),
        "reaction_ms": (190, 280), "dwell_scale": (0.85, 1.15),
        "fitts_precision": (0.85, 1.15), "tremor_scale": (0.80, 1.20),
        "flick_vigor": (0.95, 1.15), "overshoot_rate": (0.05, 0.20),
        "fling_bias": (0.40, 0.65), "patience": (0.40, 0.70), "curiosity": (0.30, 0.65),
    },
    # Unsteady / hesitant: shaky finger (high tremor + scatter), slow, imprecise
    # (overshoots, low precision), very patient, more typos.
    "shaky": {
        "wpm": (20, 34), "typo_rate": (0.08, 0.16), "burstiness": (1.0, 1.3),
        "reaction_ms": (260, 360), "dwell_scale": (1.10, 1.55),
        "fitts_precision": (0.60, 0.85), "scatter_px": (4.0, 7.0),
        "tremor_scale": (1.25, 1.75), "flick_vigor": (0.80, 1.00),
        "overshoot_rate": (0.15, 0.35), "fling_bias": (0.10, 0.30),
        "patience": (0.55, 0.90), "curiosity": (0.20, 0.55),
    },
    # Restless / distracted explorer: mixes hard flings, jumpy rhythm, overshoots,
    # very curious (wanders), impatient.
    "restless": {
        "wpm": (34, 52), "typo_rate": (0.04, 0.12), "burstiness": (1.7, 2.4),
        "reaction_ms": (160, 250), "dwell_scale": (0.75, 1.05),
        "fitts_precision": (0.75, 1.05), "tremor_scale": (0.70, 1.15),
        "flick_vigor": (1.00, 1.28), "overshoot_rate": (0.15, 0.32),
        "fling_bias": (0.55, 0.85), "patience": (0.20, 0.50), "curiosity": (0.60, 0.95),
    },
}

ARCHETYPE_NAMES = list(ARCHETYPES.keys())


def generate_persona(seed=None, archetype=None) -> BiometricPersona:
    """Build the one, exact human for this seed and archetype.
       seed=None      -> a fresh random seed.
       archetype=None -> the type is derived from the seed (so a cast of seeds
                         naturally spans all types); pass a name from
                         ARCHETYPE_NAMES (or 'random') to force the type."""
    if seed is None:
        seed = random.randrange(1000, 9_999_999)
    seed = int(seed)
    rng = random.Random(seed)

    if archetype in (None, "", "auto"):
        archetype = ARCHETYPE_NAMES[seed % len(ARCHETYPE_NAMES)]
    elif archetype == "random":
        archetype = rng.choice(ARCHETYPE_NAMES)
    elif archetype not in ARCHETYPES:
        archetype = "average"

    ranges = dict(_BASE)
    ranges.update(ARCHETYPES[archetype])

    def s(key):
        lo, hi = ranges[key]
        return rng.uniform(lo, hi)

    handed = rng.choice([-1, 1])
    return BiometricPersona(
        name="%s-%04d" % (archetype, seed % 10000),
        seed=seed,
        archetype=archetype,
        wpm=s("wpm"),
        typo_rate=s("typo_rate"),
        burstiness=s("burstiness"),
        scatter_px=s("scatter_px"),
        fitts_precision=s("fitts_precision"),
        reaction_ms=s("reaction_ms"),
        dwell_scale=s("dwell_scale"),
        curve_bias=handed * s("curve_bias"),
        tremor_scale=s("tremor_scale"),
        flick_vigor=s("flick_vigor"),
        overshoot_rate=s("overshoot_rate"),
        fatigue_rate=s("fatigue_rate"),
        fling_bias=s("fling_bias"),
        curiosity=s("curiosity"),
        patience=s("patience"),
    )


def generate_random_persona() -> BiometricPersona:
    return generate_persona(None)