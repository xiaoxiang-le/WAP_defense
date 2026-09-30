import argparse
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "AiWaf-1" / "code"
sys.path.insert(0, str(CODE_DIR))

from geturl import sniff_requests
from train_url import Train, split_word
from type import find_type


class TestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        return


def parse_args():
    parser = argparse.ArgumentParser(
        description="验证 AiWaf-1 的 HTTP 抓包和 XSS 检测链路"
    )
    parser.add_argument(
        "--interface",
        default=r"\Device\NPF_Loopback",
        help="Npcap 抓包接口，默认使用回环接口",
    )
    parser.add_argument("--port", type=int, default=18081, help="本地测试端口")
    parser.add_argument("--timeout", type=int, default=8, help="抓包超时时间")
    return parser.parse_args()


def run_capture_test(interface, port, timeout):
    server = HTTPServer(("127.0.0.1", port), TestHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    request_error = []

    def send_request():
        try:
            url = (
                "http://127.0.0.1:{}/"
                "?q=%3Cscript%3Ealert%281%29%3C/script%3E"
            ).format(port)
            urllib.request.urlopen(url, timeout=5).read()
        except Exception as error:
            request_error.append(error)

    sender = threading.Timer(1, send_request)
    sender.start()
    try:
        records = sniff_requests(
            interface=interface,
            port=port,
            timeout=timeout,
            count=1,
        )
    finally:
        sender.join(timeout=6)
        server.shutdown()
        server.server_close()

    if request_error:
        raise RuntimeError("本地 HTTP 请求失败") from request_error[0]
    if len(records) != 1:
        raise RuntimeError("预期捕获 1 条 HTTP 请求，实际捕获 {} 条".format(len(records)))

    captured_url = records[0]["url"]
    model_label = int(Train.load_or_train().predict_labels([captured_url])[0])
    risk = find_type(split_word(captured_url))
    if model_label != 1 or "XSS" not in risk:
        raise RuntimeError(
            "检测结果不符合预期：label={}，risk={}".format(model_label, risk)
        )

    print("捕获 URL：{}".format(captured_url))
    print("模型分类：恶意请求")
    print("风险判断：{}".format(risk))
    print("端到端抓包检测：通过")


def main():
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
    args = parse_args()
    run_capture_test(args.interface, args.port, args.timeout)


if __name__ == "__main__":
    main()
