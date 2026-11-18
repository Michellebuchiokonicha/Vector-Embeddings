"""
Words as Vectors — where does "queen" live in meaning-space?
Created by Michelle Buchi Okonicha

Loads real 50-dimensional GloVe embeddings, finds the nearest neighbours of a
word, projects everything down to 3D and animates the words as arrows that
start at the origin. Words with similar meanings point in similar directions.

Setup (once, inside a Python 3.12 venv):
    python3.12 -m venv .venv && source .venv/bin/activate
    pip install manim gensim numpy
    brew install ffmpeg pango

Render (each colour theme is its own scene, so pick one):
    manim -pql word_embeddings.py NavyCoral       # quick low-quality preview
    manim -pqh word_embeddings.py NavyCoral       # 1080p final
    manim -ql  word_embeddings.py NavyCoral PurpleNeon ForestGold CreamInk   # all four
"""

import wave
from pathlib import Path

import numpy as np
import gensim.downloader
from manim import *

SOUND = "both"          # "forest", "chess", "both", or None for silence
CREATOR = "Michelle Buchi Okonicha"
WORD = "queen"          # the word in the spotlight
N_NEIGHBOURS = 10       # nearest neighbours to draw
N_LABELLED = 4         # how many of them get a floating 3D label (keeps it uncluttered)
ARROW_SCALE = 3.6       # length of the longest arrow, in scene units

SANS = "Avenir Next"
MONO = "Menlo"

# Colour themes: one dict per look. Every colour in the scene comes from here.
THEMES = {
    "navy": dict(   # deep navy, coral spotlight, teal neighbours, gold accents
        bg="#0B1330", panel="#141E45", grid="#33437A", axis="#8FA3D9",
        text="#EAF0FF", muted="#8FA3D9",
        main="#FF6F61", neighbour="#3EC6C1", accent="#F4C95D",
    ),
    "purple": dict(  # deep purple, neon magenta spotlight, lilac neighbours, cyan accents
        bg="#170A2B", panel="#241241", grid="#4B2C73", axis="#B794F4",
        text="#F5EEFF", muted="#A98BD6",
        main="#FF3CAC", neighbour="#C084FC", accent="#00E5FF",
    ),
    "forest": dict(  # dark forest, gold spotlight, sage neighbours, cream accents
        bg="#0E2218", panel="#163324", grid="#2F5A43", axis="#9CC5A1",
        text="#F1F5E9", muted="#9CC5A1",
        main="#E9B949", neighbour="#7FB685", accent="#F2E8CF",
    ),
    "cream": dict(   # light paper, orange spotlight, ink-blue neighbours, brown accents
        bg="#FBF6EC", panel="#F1E7D3", grid="#D9CBB0", axis="#5B4A3A",
        text="#2B2118", muted="#8A7660",
        main="#F2662C", neighbour="#2E6F8E", accent="#9C5B1F",
    ),
}


# 1. Real embeddings, squashed from 50 dimensions down to 3
def load_vectors():
    model = gensim.downloader.load("glove-wiki-gigaword-50")
    similar = model.most_similar(WORD, topn=N_NEIGHBOURS)
    words = [WORD] + [w for w, _ in similar]
    sims = [s for _, s in similar]
    vecs = np.array([model[w] for w in words])

    # Uncentred SVD (not PCA). PCA subtracts the mean, which would drop the
    # spotlight word near the origin. Without centring, similar words keep
    # pointing roughly the same way.
    _, _, vt = np.linalg.svd(vecs, full_matrices=False)
    pts = vecs @ vt[:3].T

    # SVD axis signs are arbitrary: flip so the spotlight word points
    # forward-right and upward, above the ground rings.
    if pts[0, 0] < 0:
        pts[:, 0] *= -1
    if pts[0, 2] < 0:
        pts[:, 2] *= -1

    pts *= ARROW_SCALE / np.linalg.norm(pts, axis=1).max()
    return words, sims, pts, model[WORD]


# ---------------------------------------------------------------------------
# Sound, synthesised with numpy: no audio files to download
# ---------------------------------------------------------------------------
SR = 44100                                   # samples per second
SOUND_DIR = Path(__file__).parent / "media" / "sounds"


def write_wav(name, samples):
    SOUND_DIR.mkdir(parents=True, exist_ok=True)
    path = SOUND_DIR / name
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return str(path)


def wood_click(pitch=1.0, seed=0):
    """A chess piece set down on a wooden board: a sharp tick + short woody ring."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(0.2 * SR)) / SR
    tick = rng.normal(size=t.size) * np.exp(-t / 0.002)
    body = (0.6 * np.sin(TAU * 1050 * pitch * t) * np.exp(-t / 0.018)
            + 0.3 * np.sin(TAU * 2300 * pitch * t) * np.exp(-t / 0.008)
            + 0.5 * np.sin(TAU * 190 * pitch * t) * np.exp(-t / 0.035))
    s = body + 0.4 * tick
    return 0.8 * s / np.abs(s).max()


def bird_call(rng):
    """2-5 quick chirps from one bird, each a fast pitch sweep with a flutter."""
    base = rng.uniform(2800, 4200)
    out = []
    for _ in range(rng.integers(2, 6)):
        dur = rng.uniform(0.06, 0.14)
        t = np.arange(int(dur * SR)) / SR
        freq = base + rng.uniform(-1500, 1500) * t / dur + 150 * np.sin(TAU * 30 * t)
        phase = TAU * np.cumsum(freq) / SR
        env = np.sin(np.pi * t / dur) ** 2
        out += [env * (np.sin(phase) + 0.3 * np.sin(2 * phase)),
                np.zeros(int(rng.uniform(0.04, 0.12) * SR))]
    return np.concatenate(out)


def whistle(rng):
    """A slower two-note 'fee-bee' whistle, from a second kind of bird."""
    f1 = rng.uniform(2900, 3300)
    notes = []
    for f, dur in [(f1, 0.32), (f1 * 0.85, 0.38)]:
        t = np.arange(int(dur * SR)) / SR
        notes += [np.sin(TAU * f * t) * np.sin(np.pi * t / dur) ** 1.5,
                  np.zeros(int(0.06 * SR))]
    return np.concatenate(notes)


def forest(seconds, seed=7):
    """Wind gusts + leaf rustle + randomly placed bird calls."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR

    # Wind: white noise smoothed (low-passed) twice, swelling in slow gusts
    white = rng.normal(size=n)
    kernel = np.ones(400) / 400
    wind = np.convolve(np.convolve(white, kernel, "same"), kernel, "same")
    wind /= np.abs(wind).max()
    gusts = 0.6 + 0.4 * np.sin(TAU * t / 7 + 1.3) * np.sin(TAU * t / 11)
    rustle = (white - np.convolve(white, kernel, "same")) * 0.015
    mix = 0.12 * wind * gusts + rustle * gusts

    # Birds: a call every 1-3 s, at random volumes so some sound far away
    pos = rng.uniform(0.3, 1.2)
    while pos < seconds - 1:
        call = whistle(rng) if rng.random() < 0.25 else bird_call(rng)
        i = int(pos * SR)
        j = min(i + call.size, n)
        mix[i:j] += rng.uniform(0.08, 0.25) * call[: j - i]
        pos += rng.uniform(1.0, 3.0)

    # Fade in and out so it doesn't start or stop abruptly
    fade = np.ones(n)
    fin, fout = int(1.5 * SR), int(2.0 * SR)
    fade[:fin] = np.linspace(0, 1, fin)
    fade[-fout:] = np.linspace(1, 0, fout)
    return 0.7 * mix * fade / np.abs(mix).max()


# ---------------------------------------------------------------------------
# 2. The scene (theme-agnostic). Subclasses at the bottom pick the theme.
# ---------------------------------------------------------------------------
class EmbeddingScene(ThreeDScene):
    THEME = "navy"

    def construct(self):
        T = self.T = THEMES[self.THEME]
        self.camera.background_color = T["bg"]
        words, sims, pts, raw = load_vectors()

        # Angled so the arrows fan out to the left, away from the side panel
        self.set_camera_orientation(phi=68 * DEGREES, theta=60 * DEGREES, zoom=0.8)

        # --- 01 · Load the embeddings ----------------------------------------
        header = self.section_header("01", "Load the embeddings")
        code = self.code_panel(raw).next_to(header, DOWN, aligned_edge=LEFT, buff=0.3)
        self.fixed(header, code)

        ground, axes = self.build_space()
        self.play(Create(ground), Create(axes), run_time=1.5)
        self.play(FadeIn(header), Write(code), run_time=1.5)
        self.wait(1)

        # --- 02 · Spotlight word, then its neighbours -------------------------
        header = self.swap_header(header, self.section_header(
            "02", f"Find the {N_NEIGHBOURS} nearest neighbours of “{WORD}”"))

        origin = axes.c2p(0, 0, 0)
        main_end = axes.c2p(*pts[0])
        main_arrow = Arrow3D(start=origin, end=main_end, color=T["main"],
                             thickness=0.028, height=0.35, base_radius=0.1, resolution=12)
        main_label = Text(WORD, font=SANS, font_size=40, color=T["main"], weight=BOLD)
        main_label.move_to(main_end + 0.4 * normalize(main_end) + 1.0 * OUT)

        arrows, labels = VGroup(), VGroup()
        for i, (w, p) in enumerate(zip(words[1:], pts[1:])):
            col = self.rank_color(i)
            end = axes.c2p(*p)
            arrow = Arrow3D(start=origin, end=end, color=col, thickness=0.01,
                            height=0.22, base_radius=0.05, resolution=8)
            arrows.add(arrow.set_opacity(0.6))
            # Label the closest few, skipping any that would sit on top of
            # an existing label (e.g. "lady" and "elizabeth" nearly coincide)
            spot = end + 0.45 * normalize(end)
            too_close = any(np.linalg.norm(spot - l.get_center()) < 0.8
                            for l in [main_label, *labels])
            if len(labels) < N_LABELLED and not too_close:
                lbl = Text(w, font=SANS, font_size=24, color=col)
                labels.add(lbl.move_to(spot))

        # Labels always face the camera, even while it orbits
        for lbl in [*labels, main_label]:
            lbl.set_opacity(0)
        self.add_fixed_orientation_mobjects(*labels, main_label)

        chess = SOUND in ("chess", "both")
        if chess:
            # Deeper "thunk" for queen, landing as its arrow finishes growing
            self.add_sound(write_wav("click_main.wav", wood_click(pitch=0.7)), time_offset=1.2)
        self.play(GrowFromPoint(main_arrow, origin), run_time=1.2)
        self.play(main_label.animate.set_opacity(1), run_time=0.6)

        # LaggedStart timing: arrow i ends at run_time * (1 + lag*i) / (1 + lag*(n-1))
        run_time, lag, n = 2.5, 0.12, len(arrows)
        if chess:
            for i in range(n):
                click = write_wav(f"click_{i}.wav", wood_click(pitch=1 + 0.04 * (i % 4), seed=i))
                self.add_sound(click, time_offset=run_time * (1 + lag * i) / (1 + lag * (n - 1)),
                               gain=-4)
        self.play(
            LaggedStart(*[GrowFromPoint(a, origin) for a in arrows], lag_ratio=lag),
            LaggedStart(*[l.animate.set_opacity(0.9) for l in labels], lag_ratio=lag),
            run_time=run_time,
        )

        # Ranked similarity panel on the right
        panel, panel_bg, panel_top, rows = self.neighbour_panel(words[1:], sims)
        self.fixed(panel)
        self.play(FadeIn(panel_bg), FadeIn(panel_top), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(r, shift=0.2 * LEFT) for r in rows],
                              lag_ratio=0.12), run_time=2)
        self.wait(0.5)

        # --- 03 · Explore the space -----------------------------------------
        self.swap_header(header, self.section_header(
            "03", "Similar meaning → similar direction"))
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(8)
        self.stop_ambient_camera_rotation()
        self.wait()

        # Forest ambience: generated at the exact length of the finished
        # video, then laid in from the very start (offset back to t = 0)
        if SOUND in ("forest", "both"):
            total = self.renderer.time
            self.add_sound(write_wav("forest.wav", forest(total)), time_offset=-total, gain=-3)

    # -----------------------------------------------------------------------
    # Building blocks
    # -----------------------------------------------------------------------
    def fixed(self, *mobs):
        """Pin to the screen (ignores the 3D camera), but don't show yet."""
        self.add_fixed_in_frame_mobjects(*mobs)
        self.remove(*mobs)

    def rank_color(self, i):
        """Closest neighbour = full neighbour colour, fading towards muted."""
        alpha = 0.7 * i / max(N_NEIGHBOURS - 1, 1)
        return ManimColor(self.T["neighbour"]).interpolate(ManimColor(self.T["muted"]), alpha)

    def section_header(self, num, title):
        T = self.T
        tag = Text(num, font=MONO, font_size=18, color=T["bg"], weight=BOLD)
        pill = RoundedRectangle(corner_radius=0.15, width=tag.width + 0.35,
                                height=tag.height + 0.22, fill_color=T["accent"],
                                fill_opacity=1, stroke_width=0)
        tag.move_to(pill)
        text = Text(title, font=SANS, font_size=28, color=T["text"], weight=BOLD)
        return VGroup(VGroup(pill, tag), text).arrange(RIGHT, buff=0.25).to_corner(UL, buff=0.4)

    def swap_header(self, old, new):
        self.fixed(new)
        self.play(FadeOut(old, shift=0.3 * UP), FadeIn(new, shift=0.3 * UP), run_time=0.8)
        return new

    def code_panel(self, raw):
        T = self.T
        kw, st = T["accent"], T["neighbour"]
        lines = VGroup(
            Text(f"# word embeddings, by {CREATOR}", font=MONO, font_size=18,
                 color=T["muted"], t2c={CREATOR: T["accent"]}),
            Text("import gensim.downloader as api", font=MONO, font_size=18,
                 color=T["text"], t2c={"import": kw, "as": kw}),
            Text('glove = api.load("glove-wiki-gigaword-50")', font=MONO, font_size=18,
                 color=T["text"], t2c={'"glove-wiki-gigaword-50"': st}),
            Text(f'glove["{WORD}"]', font=MONO, font_size=18,
                 color=T["text"], t2c={f'"{WORD}"': st}),
            Text(f"# → [{raw[0]:.2f}, {raw[1]:.2f}, {raw[2]:.2f}, …]  50 numbers",
                 font=MONO, font_size=16, color=T["muted"]),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        bg = SurroundingRectangle(lines, buff=0.25, corner_radius=0.15,
                                  fill_color=T["panel"], fill_opacity=0.92,
                                  stroke_color=T["grid"], stroke_width=1.5)
        stripe = Line(bg.get_corner(UL) + 0.08 * RIGHT + 0.15 * DOWN,
                      bg.get_corner(DL) + 0.08 * RIGHT + 0.15 * UP,
                      color=T["main"], stroke_width=4)
        return VGroup(bg, stripe, lines)

    def build_space(self):
        """Concentric rings + spokes on the ground instead of a square grid."""
        T = self.T
        rings = VGroup(*[
            Circle(radius=r, stroke_color=T["grid"], stroke_width=1.2, stroke_opacity=0.6)
            for r in range(1, 7)
        ])
        spokes = VGroup(*[
            Line(ORIGIN, 6 * np.array([np.cos(a), np.sin(a), 0]),
                 stroke_color=T["grid"], stroke_width=1, stroke_opacity=0.35)
            for a in np.arange(0, TAU, TAU / 12)
        ])
        axes = ThreeDAxes(
            x_range=[-5, 5, 1], y_range=[-5, 5, 1], z_range=[-3.5, 3.5, 1],
            x_length=10, y_length=10, z_length=7,
            axis_config={"color": T["axis"], "stroke_width": 2,
                         "include_tip": False, "include_ticks": False},
        )
        return VGroup(rings, spokes), axes

    def neighbour_panel(self, words, sims):
        """Ranked list: rank · word · similarity bar · score."""
        T = self.T
        top = VGroup(
            Text(f"closest to “{WORD}”", font=SANS, font_size=22, color=T["accent"], weight=BOLD),
            Text("cosine similarity", font=SANS, font_size=15, color=T["muted"]),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)

        rows = VGroup()
        for i, (w, s) in enumerate(zip(words, sims)):
            rank = Text(f"{i + 1:02d}", font=MONO, font_size=15, color=T["muted"])
            word = Text(w, font=SANS, font_size=21, color=T["text"])
            bar = Rectangle(width=max(s, 0.02) * 1.4, height=0.13, stroke_width=0,
                            fill_color=self.rank_color(i), fill_opacity=1)
            val = Text(f"{s:.2f}", font=MONO, font_size=15, color=T["muted"])
            for mob, x in [(rank, 0), (word, 0.5), (bar, 2.25), (val, 3.75)]:
                mob.move_to([x, 0, 0], aligned_edge=LEFT)
            rows.add(VGroup(rank, word, bar, val))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.2)

        body = VGroup(top, rows).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        bg = SurroundingRectangle(body, buff=0.3, corner_radius=0.15,
                                  fill_color=T["panel"], fill_opacity=0.92,
                                  stroke_color=T["grid"], stroke_width=1.5)
        panel = VGroup(bg, body).scale(0.85).to_edge(RIGHT, buff=0.3)
        return panel, bg, top, rows


# ---------------------------------------------------------------------------
# 3. Theme switches: render whichever class you want
# ---------------------------------------------------------------------------
class NavyCoral(EmbeddingScene):
    THEME = "navy"


class PurpleNeon(EmbeddingScene):
    THEME = "purple"


class ForestGold(EmbeddingScene):
    THEME = "forest"


class CreamInk(EmbeddingScene):
    THEME = "cream"
