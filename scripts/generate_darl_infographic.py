"""Generate a DARL thesis infographic from the current thesis content."""

from __future__ import annotations

import shutil
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures"
THESIS_DIR = ROOT / "thesis" / "figures" / "generated"

WIDTH = 1800
HEIGHT = 3000
MARGIN = 110

BG = "#f6f8fb"
INK = "#172033"
MUTED = "#5b6472"
LINE = "#d9dee8"
TEAL = "#007c89"
BLUE = "#2f6fdd"
CORAL = "#d95f5f"
AMBER = "#e3a12d"
GREEN = "#2c9a63"
PURPLE = "#6f5cc2"
WHITE = "#ffffff"
SOFT_TEAL = "#e2f3f4"
SOFT_BLUE = "#e6efff"
SOFT_CORAL = "#fdeceb"
SOFT_AMBER = "#fff3d8"
SOFT_GREEN = "#e5f5eb"
SOFT_PURPLE = "#efebff"


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    candidates = {
        "regular": [
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arial.ttf",
        ],
        "bold": [
            r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
        ],
        "semibold": [
            r"C:\Windows\Fonts\seguisb.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf",
        ],
    }[name]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONTS = {
    "title": font("bold", 116),
    "subtitle": font("regular", 40),
    "section": font("bold", 42),
    "card_title": font("bold", 34),
    "body": font("regular", 28),
    "small": font("regular", 23),
    "tiny": font("regular", 20),
    "metric": font("bold", 48),
    "metric_label": font("semibold", 24),
}


def text_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    max_width: int,
    font_obj: ImageFont.ImageFont,
    fill: str = INK,
    line_gap: int = 8,
    bullet: bool = False,
) -> int:
    x, y = xy
    avg_char = max(8, int(font_obj.size * 0.52))
    max_chars = max(18, max_width // avg_char)
    lines: list[str] = []
    for paragraph in text.split("\n"):
        prefix = "• " if bullet and paragraph else ""
        wrapped = wrap(paragraph, width=max_chars)
        if not wrapped:
            lines.append("")
            continue
        for i, line in enumerate(wrapped):
            lines.append((prefix if i == 0 else "  ") + line)
    for line in lines:
        draw.text((x, y), line, font=font_obj, fill=fill)
        y += font_obj.size + line_gap
    return y


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str | None = None,
    width: int = 2,
    radius: int = 26,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    body: str,
    accent: str,
    fill: str = WHITE,
    title_fill: str = INK,
    body_fill: str = MUTED,
) -> None:
    x1, y1, x2, y2 = box
    rounded(draw, box, fill=fill, outline=LINE, radius=24)
    draw.rounded_rectangle((x1, y1, x1 + 16, y2), radius=24, fill=accent)
    draw.text((x1 + 42, y1 + 30), title, font=FONTS["card_title"], fill=title_fill)
    text_box(draw, (x1 + 42, y1 + 82), body, x2 - x1 - 80, FONTS["body"], body_fill)


def metric_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    label: str,
    accent: str,
    fill: str,
) -> None:
    x1, y1, x2, y2 = box
    rounded(draw, box, fill=fill, outline=LINE, radius=24)
    draw.text((x1 + 34, y1 + 26), value, font=FONTS["metric"], fill=accent)
    text_box(draw, (x1 + 34, y1 + 88), label, x2 - x1 - 68, FONTS["metric_label"], INK, line_gap=5)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str) -> None:
    draw.line((start, end), fill=color, width=8)
    sx, sy = start
    ex, ey = end
    if ex >= sx:
        head = [(ex, ey), (ex - 22, ey - 14), (ex - 22, ey + 14)]
    else:
        head = [(ex, ey), (ex + 22, ey - 14), (ex + 22, ey + 14)]
    draw.polygon(head, fill=color)


def draw_pipeline(draw: ImageDraw.ImageDraw, y: int) -> None:
    draw.text((MARGIN, y), "Arquitectura evaluada", font=FONTS["section"], fill=INK)
    y += 70
    boxes = [
        ((MARGIN, y, MARGIN + 310, y + 180), "Datos tabulares", "TableShift\nPhysioNet\ndiabetes", TEAL, SOFT_TEAL),
        ((MARGIN + 395, y, MARGIN + 735, y + 180), "Stage 1", "Preprocesamiento\nf_phi", BLUE, SOFT_BLUE),
        ((MARGIN + 820, y, MARGIN + 1160, y + 180), "Stage 2", "Modelo predictivo\ng_theta", PURPLE, SOFT_PURPLE),
        ((MARGIN + 1245, y, WIDTH - MARGIN, y + 180), "Decisión", "AUC, costo y\nacción selectiva", GREEN, SOFT_GREEN),
    ]
    for box, title, body, color, fill in boxes:
        rounded(draw, box, fill=fill, outline=color, width=3, radius=24)
        x1, y1, x2, _ = box
        draw.text((x1 + 24, y1 + 22), title, font=FONTS["card_title"], fill=color)
        text_box(draw, (x1 + 24, y1 + 72), body, x2 - x1 - 48, FONTS["body"], INK, line_gap=4)
    for x in [MARGIN + 330, MARGIN + 755, MARGIN + 1180]:
        arrow(draw, (x, y + 90), (x + 48, y + 90), MUTED)


def draw_cycle(draw: ImageDraw.ImageDraw, y: int) -> None:
    draw.text((MARGIN, y), "Ciclo DARL como POMDP", font=FONTS["section"], fill=INK)
    y += 70
    x_center = WIDTH // 2
    nodes = [
        ((MARGIN, y + 60, MARGIN + 330, y + 190), "Drift sintético", "covariate, concept o ambos", CORAL, SOFT_CORAL),
        ((MARGIN + 410, y, MARGIN + 750, y + 130), "Monitor", "PSI, KS, C2ST, ΔAUC, F1", TEAL, SOFT_TEAL),
        ((MARGIN + 835, y, MARGIN + 1195, y + 130), "Agente PPO", "observa señales parciales", PURPLE, SOFT_PURPLE),
        ((WIDTH - MARGIN - 330, y + 60, WIDTH - MARGIN, y + 190), "Acción", "diferir o actualizar etapa", GREEN, SOFT_GREEN),
        ((x_center - 190, y + 245, x_center + 190, y + 375), "Reward", "recuperar AUC con bajo costo", AMBER, SOFT_AMBER),
    ]
    for box, title, body, color, fill in nodes:
        rounded(draw, box, fill=fill, outline=color, width=3, radius=24)
        x1, y1, x2, _ = box
        draw.text((x1 + 24, y1 + 20), title, font=FONTS["card_title"], fill=color)
        text_box(draw, (x1 + 24, y1 + 70), body, x2 - x1 - 48, FONTS["small"], INK, line_gap=4)
    arrow(draw, (MARGIN + 330, y + 125), (MARGIN + 400, y + 80), MUTED)
    arrow(draw, (MARGIN + 750, y + 65), (MARGIN + 825, y + 65), MUTED)
    arrow(draw, (MARGIN + 1195, y + 65), (WIDTH - MARGIN - 340, y + 125), MUTED)
    arrow(draw, (WIDTH - MARGIN - 170, y + 200), (x_center + 190, y + 305), MUTED)
    arrow(draw, (x_center - 190, y + 305), (MARGIN + 175, y + 200), MUTED)


def build() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((0, 0, WIDTH, 430), radius=0, fill=INK)
    draw.text((MARGIN, 78), "DARL", font=FONTS["title"], fill=WHITE)
    draw.text(
        (MARGIN, 205),
        "Aprendizaje por refuerzo con diagnóstico de drift para actualizar pipelines tabulares de ML",
        font=FONTS["subtitle"],
        fill="#dfe7f3",
    )
    draw.text(
        (MARGIN, 284),
        "Idea central: decidir qué componente actualizar, no reentrenar todo por defecto.",
        font=FONTS["section"],
        fill=AMBER,
    )
    draw.text((MARGIN, 366), "Basado en thesis/build/main.pdf - tesis DARL, 2026", font=FONTS["small"], fill="#aeb8c8")

    card(
        draw,
        (MARGIN, 500, WIDTH // 2 - 35, 760),
        "Problema",
        "El rendimiento cae cuando cambia P(X) o P(Y|X). En pipelines de dos etapas, la falla puede estar en el preprocesamiento, en el modelo o en ambos.",
        CORAL,
        SOFT_CORAL,
    )
    card(
        draw,
        (WIDTH // 2 + 35, 500, WIDTH - MARGIN, 760),
        "Pregunta de investigación",
        "¿Puede una política de RL seleccionar la actualización selectiva que maximiza recuperación predictiva y minimiza costo computacional ante drift no visto?",
        TEAL,
        SOFT_TEAL,
    )

    draw_pipeline(draw, 850)

    y = 1165
    draw.text((MARGIN, y), "Señales de diagnóstico", font=FONTS["section"], fill=INK)
    y += 66
    signal_boxes = [
        ((MARGIN, y, MARGIN + 360, y + 190), "Covariate shift", "PSI\nKS\nC2ST sobre X", TEAL, SOFT_TEAL),
        ((MARGIN + 400, y, MARGIN + 760, y + 190), "Concept drift", "ΔAUC\nΔloss\nF1 y errores", CORAL, SOFT_CORAL),
        ((MARGIN + 800, y, MARGIN + 1160, y + 190), "Severidad", "leve\nmoderada\nsevera", AMBER, SOFT_AMBER),
        ((MARGIN + 1200, y, WIDTH - MARGIN, y + 190), "Observación", "vector o_t\npara PPO", PURPLE, SOFT_PURPLE),
    ]
    for box, title, body, color, fill in signal_boxes:
        rounded(draw, box, fill=fill, outline=color, width=3, radius=24)
        x1, y1, x2, _ = box
        draw.text((x1 + 24, y1 + 20), title, font=FONTS["card_title"], fill=color)
        text_box(draw, (x1 + 24, y1 + 70), body, x2 - x1 - 48, FONTS["body"], INK, line_gap=3)

    draw_cycle(draw, 1505)

    y = 2000
    draw.text((MARGIN, y), "Acciones de mantenimiento", font=FONTS["section"], fill=INK)
    y += 70
    action_w = (WIDTH - 2 * MARGIN - 3 * 28) // 4
    actions = [
        ("Defer", "No intervenir si el costo no se justifica.", MUTED, WHITE),
        ("Update features", "Reajustar Stage 1: f_phi.", BLUE, SOFT_BLUE),
        ("Update model", "Reentrenar Stage 2: g_theta.", PURPLE, SOFT_PURPLE),
        ("Retrain all", "Actualizar preprocesamiento y modelo.", GREEN, SOFT_GREEN),
    ]
    for i, (title, body, color, fill) in enumerate(actions):
        x = MARGIN + i * (action_w + 28)
        rounded(draw, (x, y, x + action_w, y + 170), fill=fill, outline=color, width=3, radius=24)
        draw.text((x + 24, y + 24), title, font=FONTS["card_title"], fill=color)
        text_box(draw, (x + 24, y + 78), body, action_w - 48, FONTS["small"], INK, line_gap=5)

    y = 2310
    draw.text((MARGIN, y), "Resultados preliminares reportados", font=FONTS["section"], fill=INK)
    y += 72
    metric_w = (WIDTH - 2 * MARGIN - 3 * 28) // 4
    metrics = [
        ("0.7836", "AUC base XGBoost antes del drift", BLUE, SOFT_BLUE),
        ("5000", "decisiones en evaluación PPO", PURPLE, SOFT_PURPLE),
        ("0.0334", "reward medio por paso", GREEN, SOFT_GREEN),
        ("59.86%", "selección de Update features", TEAL, SOFT_TEAL),
    ]
    for i, (value, label, color, fill) in enumerate(metrics):
        x = MARGIN + i * (metric_w + 28)
        metric_card(draw, (x, y, x + metric_w, y + 165), value, label, color, fill)

    card(
        draw,
        (MARGIN, 2680, WIDTH - MARGIN, 2875),
        "Lectura preliminar",
        "La acción óptima cambia con tipo y severidad de drift; el costo computacional puede cambiar la decisión incluso cuando una estrategia tiene mayor AUC absoluto. La demo aún valida plausibilidad, no convergencia definitiva.",
        AMBER,
        SOFT_AMBER,
    )

    draw.text(
        (MARGIN, HEIGHT - 58),
        "Fuente: DARL, main.pdf. Métricas tomadas de resultados preliminares sobre PhysioNet y demo PPO.",
        font=FONTS["tiny"],
        fill=MUTED,
    )
    return image


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_DIR.mkdir(parents=True, exist_ok=True)

    image = build()
    png = OUT_DIR / "darl_infografia.png"
    pdf = OUT_DIR / "darl_infografia.pdf"
    image.save(png, optimize=True)
    image.save(pdf, "PDF", resolution=300.0)

    shutil.copy2(png, THESIS_DIR / png.name)
    shutil.copy2(pdf, THESIS_DIR / pdf.name)
    print(png)
    print(pdf)
    print(THESIS_DIR / png.name)
    print(THESIS_DIR / pdf.name)


if __name__ == "__main__":
    main()
