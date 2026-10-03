import argparse
import json
import sys

from train_url import Train


def configure_console_encoding():
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description="AiWaf-1 HTTP 实时入侵检测界面")
    parser.add_argument("--interface", help="抓包网卡名称；不指定时使用系统默认网卡")
    parser.add_argument("--port", type=int, default=80, help="监听的 HTTP TCP 端口")
    parser.add_argument(
        "--max-body-bytes",
        type=int,
        default=65536,
        help="最多读取的 HTTP 请求体字节数，默认 65536",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="重新训练 AiWaf-1 分类模型后退出",
    )
    parser.add_argument("--log-file", help="将检测事件追加写入指定 JSONL 文件")
    parser.add_argument(
        "--log-body",
        action="store_true",
        help="在 JSONL 日志中记录请求体；可能包含敏感信息",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="判定恶意请求的概率阈值，范围 0 到 1，默认 0.5",
    )
    parser.add_argument(
        "--log-max-bytes",
        type=int,
        default=10 * 1024 * 1024,
        help="单个日志文件的大小上限，0 表示不轮转，默认 10485760",
    )
    parser.add_argument(
        "--log-backups",
        type=int,
        default=5,
        help="轮转日志的保留数量，默认 5",
    )
    parser.add_argument(
        "--payload",
        action="append",
        help="离线检测一条 URL 或 Payload；可重复指定以批量检测",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="将离线检测结果输出为 JSON",
    )
    args = parser.parse_args(arguments)
    if args.log_body and not args.log_file:
        parser.error("--log-body 必须与 --log-file 一起使用")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold 必须在 0 到 1 之间")
    if args.log_max_bytes < 0:
        parser.error("--log-max-bytes 不能为负数")
    if args.log_backups < 0:
        parser.error("--log-backups 不能为负数")
    if args.retrain and args.payload:
        parser.error("--retrain 不能与 --payload 同时使用")
    if args.json and not args.payload:
        parser.error("--json 必须与 --payload 一起使用")
    return args


def detect_payloads(payloads, threshold=0.5, json_output=False):
    """使用 AiWaf-1 模型离线检测 Payload，并将结果写入标准输出。"""
    detector = Train.load_or_train()
    details = detector.predict_details(payloads, threshold=threshold)
    results = [
        {"payload": payload, **detail}
        for payload, detail in zip(payloads, details)
    ]
    if json_output:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(
                "{}  恶意概率：{:.2%}  Payload：{}".format(
                    result["message"],
                    result["malicious_probability"],
                    result["payload"],
                )
            )
    return results


def main():
    configure_console_encoding()
    args = parse_args()
    if args.retrain:
        Train().model_train()
        return
    if args.payload:
        detect_payloads(args.payload, threshold=args.threshold, json_output=args.json)
        return
    from UI import UI_start

    if not 1 <= args.port <= 65535:
        raise SystemExit("--port 必须在 1 到 65535 之间")
    if args.max_body_bytes < 0:
        raise SystemExit("--max-body-bytes 不能为负数")
    UI_start(
        interface=args.interface,
        port=args.port,
        max_body_bytes=args.max_body_bytes,
        log_file=args.log_file,
        log_body=args.log_body,
        threshold=args.threshold,
        log_max_bytes=args.log_max_bytes,
        log_backups=args.log_backups,
    )


if __name__ == "__main__":
    main()
