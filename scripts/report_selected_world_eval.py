# 中文：正式入口：把已保存的选中动作重建结果汇总为 HTML 报告。
# English: Public entry: render saved selected-action world predictions as an HTML report.
# 调用 / Invocation: 离线读取评测产物并写报告，不运行模型。 / Reads saved eval artifacts and writes reports; no model inference.
# 导航 / Guide: scripts/README.md (public / 正式入口)
"""Build an offline HTML report for selected-action world-model eval artifacts."""

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw


def composite(paths, size):
    if not all(path.exists() for path in paths):
        return None
    width, height = size
    main = width * 2 // 3
    image = Image.new("RGB", size)
    for path, xy, wh in zip(paths, [(0, 0), (main, 0), (main, height // 2)],
                            [(main, height), (width - main, height // 2),
                             (width - main, height - height // 2)], strict=True):
        with Image.open(path) as source:
            image.paste(source.convert("RGB").resize(wh), xy)
    return image


def build_report(root):
    root = Path(root)
    body = ['<h1>Q argmax: selected-action world predictions</h1>',
            '<p>Each row: actual input at t, predicted endpoint t+48, actual endpoint. '
            'Blue bars show all 48 expected rewards (range -1 to 0). '
            'Terminal refers to predicted success, not an environment timeout.</p>']
    summary = []
    for directory in sorted(root.glob("*/seed_*")):
        files = sorted(directory.glob("step_*.json"))
        if not files:
            continue
        rows = [json.loads(path.read_text()) for path in files]
        outcome_path = directory / "outcome.json"
        outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else {}
        success = outcome.get("success")
        title = f"{directory.parent.name} / {directory.name} / success={success}"
        body.append(f'<h2>{html.escape(title)}</h2>')
        snapshots = []
        for index, (path, row) in enumerate(zip(files, rows, strict=True)):
            stem = path.stem
            predicted_path = directory / row["image"]
            with Image.open(predicted_path) as source:
                predicted = source.convert("RGB")
            current = composite([directory / f"{stem}_{name}.png" for name in
                                 ("cam_high", "cam_left_wrist", "cam_right_wrist")], predicted.size)
            endpoint_step = row["step"] + row["horizon"]
            next_row = next((r for r in rows if r["step"] == endpoint_step), None)
            endpoint_label = "actual t+48"
            if next_row:
                actual = composite([directory / f"step_{endpoint_step:04d}_{name}.png" for name in
                                    ("cam_high", "cam_left_wrist", "cam_right_wrist")], predicted.size)
            elif index == len(rows) - 1 and outcome:
                actual = composite([directory / f"actual_final_{name}.png" for name in
                                    ("head_camera", "left_camera", "right_camera")], predicted.size)
                endpoint_label = "actual episode end (may be before t+48)"
            else:
                actual = None
                endpoint_label = "endpoint unavailable"
            caption = (f"step={row['step']} | Q={row['selected_q']:.2f} | "
                       f"R48={row['chunk_return']:.2f} | "
                       f"P(success)={row['success_probability']:.4f} | "
                       f"terminal={row['predicted_terminal']}")
            canvas = Image.new("RGB", (1152, 246), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 6), caption, fill="black")
            for j, (label, frame) in enumerate(zip(
                ("actual input t", "world prediction t+48", endpoint_label),
                (current, predicted, actual), strict=True
            )):
                draw.text((j * 384 + 8, 27), label, fill="black")
                if frame is not None:
                    canvas.paste(frame.resize((384, 192)), (j * 384, 48))
            panel = directory / f"{stem}_comparison.jpg"
            canvas.save(panel, quality=90)
            bars = ''.join(f'<rect x="{i*10}" y="{40*(1+r):.3f}" width="8" '
                           f'height="{-40*r:.3f}" fill="#4285c5"/>'
                           for i, r in enumerate(row["rewards"]))
            rel = panel.relative_to(root).as_posix()
            data_rel = path.relative_to(root).as_posix()
            body.append(f'<article><img src="{rel}"><br><svg viewBox="0 0 480 42" '
                        f'width="480" height="42">{bars}</svg> '
                        f'<a href="{data_rel}">Full 48-step reward / logits / action JSON</a></article>')
            if index in {0, len(rows)//2, len(rows)-1}:
                snapshots.append(canvas)
        overview = Image.new("RGB", (1152, 246 * len(snapshots)), "white")
        for i, panel in enumerate(snapshots):
            overview.paste(panel, (0, i * 246))
        overview.save(directory / "overview.jpg", quality=92)
        summary.append(dict(seed=rows[0]["seed"], success=success, chunks=len(rows),
                            last_q=rows[-1]["selected_q"],
                            last_success_probability=rows[-1]["success_probability"],
                            last_chunk_return=rows[-1]["chunk_return"],
                            predicted_terminal_chunks=sum(r["predicted_terminal"] for r in rows)))
    (root / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Selected world eval</title>'
        '<style>body{font:16px sans-serif;max-width:1200px;margin:24px auto;background:#f5f6f8}'
        'article{background:white;padding:10px;margin:12px 0}img{max-width:100%}</style>'
        + '\n'.join(body), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    build_report(parser.parse_args().root)
