# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Score saved H3 preview MP4s on one GPU and export a portable HTML comparison."""

import argparse
import base64
import json
import os
import sys
from html import escape
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from serve_video_edit_reward import DEFAULT_SOURCE_ROOT

PREVIEW = Path(__file__).resolve().parents[3] / "outputs/minimax_h3_video_edit_v6/preview_sft_20261113_204445"
HQ_CHECKPOINT = Path("checkpoints/three_dim_hq_final_20261124T103616Z/train/checkpoint-116")
LABELS = {"instruction_following": "指令遵循", "source_fidelity": "源视频保真度", "visual_quality": "视觉质量"}


def _report(rows, checkpoint):
    style = """
    *{box-sizing:border-box}body{max-width:1040px;margin:0 auto;padding:24px clamp(12px,4vw,48px);
    background:#f5f4ef;color:#192e31;font:16px/1.55 system-ui,sans-serif}h1{font-size:clamp(26px,4vw,40px)}
    h2{font-size:20px}section{background:white;padding:clamp(12px,3vw,28px);margin:24px 0;border-radius:12px}
    .pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0;min-width:0}
    video{background:#162225;width:100%;height:min(50vh,500px);display:block;object-fit:contain}
    figcaption{font-weight:600;margin-bottom:4px}button{cursor:pointer;font:inherit;border:0;border-radius:5px;
    background:#165953;color:white;padding:6px 14px;margin-top:12px}.scores{display:flex;flex-wrap:wrap;gap:6px 24px}
    code{overflow-wrap:anywhere}blockquote{margin:8px 0;border-left:3px solid #94b5a9;padding-left:12px}
    """
    parts = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>H3 SFT 视频编辑评估</title>',
        f"<style>{style}</style><main><h1>H3 SFT 视频编辑评估</h1><p>{len(rows)} 组视频对照</p>",
    ]
    if checkpoint:
        parts.append(
            f"<p>Reward checkpoint <code>{escape(str(checkpoint))}</code><br>综合分为三个原始评分的等权平均。</p>"
        )
    else:
        parts.append("<p>尚未运行 reward 推理。</p>")
    for row in rows:
        task = {"keep_main": "保留主体，移除路人", "remove_all": "移除所有人"}.get(row["task"], row["task"])
        parts.append(f'<section><h2>样本 {row["row_index"]:06d} · {escape(task)}</h2><div class="pair">')
        for key, label in (("source_video", "源视频"), ("edited_video", "SFT 编辑结果")):
            data = base64.b64encode(Path(row[key]).read_bytes()).decode("ascii")
            parts.append(
                f'<figure><figcaption>{label}</figcaption><video controls playsinline preload="metadata" '
                f'src="data:video/mp4;base64,{data}"></video></figure>'
            )
        parts.append('</div><button type="button" onclick="replay(this)">同步从头播放</button>')
        if "scores" in row:
            parts.append('<p class="scores">')
            for key, label in {"score": "综合分", **LABELS}.items():
                parts.append(f"<span>{label} <strong>{row['scores'][key]:.3f}</strong></span>")
            parts.append("</p>")
        parts.append(f'<p>评分指令</p><blockquote lang="en">{escape(row["edit_instruction"])}</blockquote></section>')
    parts.append("""</main><script>function replay(button){
        for(const video of document.querySelectorAll('video'))video.pause();
        for(const video of button.closest('section').querySelectorAll('video')){video.currentTime=0;video.play();}
    }</script></html>""")
    return "\n".join(parts)


def main():
    """Read the preview manifest, score its existing clips, and publish the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, default=PREVIEW)
    parser.add_argument(
        "--reward-root", type=Path, default=Path(os.environ.get("VIDEO_REWARD_ROOT", DEFAULT_SOURCE_ROOT))
    )
    parser.add_argument("--checkpoint", type=Path, help="Default: HQ final checkpoint-116 under --reward-root.")
    parser.add_argument("--output-dir", type=Path, help="Default: reward_eval under --preview-dir.")
    parser.add_argument("--preview-only", action="store_true", help="Export the video comparison without scoring.")
    args = parser.parse_args()
    preview = args.preview_dir.resolve()
    output = args.output_dir or preview / "reward_eval"
    root = args.reward_root.resolve()
    rows = []
    for line in (preview / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        source = (preview / sample["source_preview"]).resolve(strict=True)
        edited = (preview / sample["generated_video"]).resolve(strict=True)
        rows.append(
            dict(
                row_index=sample["row_index"],
                sample_id=sample["sample_id"],
                task=sample["instruction_task"],
                source_video=str(source),
                edited_video=str(edited),
                edit_instruction=sample["edit_instruction"],
            )
        )
    if not rows or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Expected a non-empty preview manifest with unique sample IDs.")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    with TemporaryDirectory(dir=output) as temp_dir:
        stage = Path(temp_dir)
        if args.preview_only:
            report_name = "video_pairs.html"
        else:
            sys.path.insert(0, str(root))
            from video_reward.inference.api import VideoRewardScorer

            scorer = VideoRewardScorer(args.checkpoint or root / HQ_CHECKPOINT, device="cuda:0")
            checkpoint = scorer.checkpoint
            if set(scorer.dimensions) != set(LABELS):
                raise ValueError(f"Expected the three video editing reward heads, got {scorer.dimensions}")
            for index, row in enumerate(rows, 1):
                scores = scorer.score(row["source_video"], row["edited_video"], row["edit_instruction"])
                row["scores"] = {**scores, "score": mean(scores.values())}
                row["checkpoint"] = str(checkpoint)
                print(f"[{index}/{len(rows)}] {row['row_index']:06d} {row['scores']}", flush=True)
            (stage / "scores.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8"
            )
            report_name = "index.html"
        (stage / report_name).write_text(_report(rows, checkpoint), encoding="utf-8")
        for artifact in stage.iterdir():
            artifact.replace(output / artifact.name)
    print(output / report_name)


if __name__ == "__main__":
    main()
