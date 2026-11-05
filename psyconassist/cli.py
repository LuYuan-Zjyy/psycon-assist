"""命令行入口。

用法：
  python -m psyconassist.cli demo --subject P-0001 --duration 3000
  python -m psyconassist.cli run --video x.mp4 --subject P-0001 --client-sample 12.4 15.4
"""
from __future__ import annotations

import argparse
import json

from . import pipeline


def main():
    ap = argparse.ArgumentParser(description="PsyConAssist 流水线")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="合成数据跑全流程")
    d.add_argument("--subject", default="P-0001")
    d.add_argument("--duration", type=float, default=3000.0)
    d.add_argument("--seed", type=int, default=42)

    r = sub.add_parser("run", help="真实视频跑全流程")
    r.add_argument("--video", required=True)
    r.add_argument("--subject", default="P-0001")
    r.add_argument("--client-sample", type=float, nargs=2, required=True,
                   help="来访者声音样本时间段 [t0 t1]，秒")

    args = ap.parse_args()
    if args.cmd == "demo":
        result = pipeline.run_synthetic(args.subject, args.duration, args.seed)
    else:
        result = pipeline.run_full(args.video, args.client_sample, args.subject)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
