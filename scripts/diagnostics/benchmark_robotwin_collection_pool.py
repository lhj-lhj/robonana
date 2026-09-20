#!/usr/bin/env python3
# 中文：旧诊断命令仅转发到统一评测组件；不维护第二套仿真或推理循环。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "internal"))
from robotwin_eval_pool import main, server_command
if __name__ == "__main__":
    main()
