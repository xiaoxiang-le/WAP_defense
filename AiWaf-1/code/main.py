import argparse
import sys

from train_url import Train


def configure_console_encoding():
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="AiWaf-1 HTTP 实时入侵检测界面")
    parser.add_argument("--interface", help="抓包网卡名称；不指定时使用系统默认网卡")
    parser.add_argument("--port", type=int, default=80, help="监听的 HTTP TCP 端口")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="重新训练 AiWaf-1 分类模型后退出",
    )
    return parser.parse_args()


def main():
    configure_console_encoding()
    args = parse_args()
    if args.retrain:
        Train().model_train()
        return
    from UI import UI_start

    UI_start(interface=args.interface, port=args.port)


if __name__ == "__main__":
    main()
