import queue
import threading
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

from detection_log import JsonlDetectionLogger
from geturl import sniff_requests
from train_url import Train, split_word
from type import find_type


BASE_DIR = Path(__file__).resolve().parent.parent


class WafUI:
    def __init__(
        self,
        root,
        interface=None,
        port=80,
        max_body_bytes=65536,
        log_file=None,
        log_body=False,
        threshold=0.5,
    ):
        self.root = root
        self.interface = interface
        self.port = port
        self.max_body_bytes = max_body_bytes
        self.logger = JsonlDetectionLogger(log_file, log_body) if log_file else None
        self.threshold = threshold
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.warning_image = None

        self._build()
        self.root.after(100, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self.exit_ui)

    def _build(self):
        self.root.title("Web 入侵检测系统")
        self.root.geometry("700x650+10+10")

        image_frame = tk.LabelFrame(self.root, text="Warning")
        image_frame.pack(fill="x", padx=10, pady=10)
        image = Image.open(BASE_DIR / "image" / "warning.jpeg")
        image.thumbnail((627, 220))
        self.warning_image = ImageTk.PhotoImage(image)
        tk.Label(image_frame, image=self.warning_image).pack()

        info_frame = tk.LabelFrame(self.root, text="运行信息")
        info_frame.pack(fill="x", padx=10, pady=5)
        interface_text = self.interface or "系统默认网卡"
        tk.Label(
            info_frame,
            text="监听接口：{}    HTTP 端口：{}    阈值：{:.0%}    日志：{}".format(
                interface_text,
                self.port,
                self.threshold,
                self.logger.path if self.logger else "关闭",
            ),
        ).pack(pady=6)

        result_frame = tk.LabelFrame(self.root, text="实时入侵检测结果")
        result_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.result_list = tk.Listbox(result_frame, width=90, height=15)
        self.result_list.pack(fill="both", expand=True, padx=6, pady=6)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=8)
        self.start_button = tk.Button(
            button_frame, text="开始", width=10, command=self.toggle_detection
        )
        self.start_button.grid(column=0, row=0, padx=5)
        tk.Button(button_frame, text="退出", width=10, command=self.exit_ui).grid(
            column=1, row=0, padx=5
        )
        tk.Button(button_frame, text="帮助", width=10, command=self.show_help).grid(
            column=2, row=0, padx=5
        )

    def toggle_detection(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.start_button.config(text="开始")
            return

        self.stop_event.clear()
        self.start_button.config(text="停止")
        self.worker = threading.Thread(target=self._capture_loop, daemon=True)
        self.worker.start()

    def _capture_loop(self):
        try:
            detector = Train.load_or_train()
            self.events.put(["检测已启动，正在等待 HTTP 请求..."])
            while not self.stop_event.is_set():
                records = sniff_requests(
                    interface=self.interface,
                    port=self.port,
                    timeout=1,
                    count=1,
                    max_body_bytes=self.max_body_bytes,
                )
                for record in records:
                    detail = detector.predict_details(
                        [record["payload"]], threshold=self.threshold
                    )[0]
                    result = detail["message"]
                    probability_text = "恶意概率：{:.2%}".format(
                        detail["malicious_probability"]
                    )
                    attack_type = (
                        find_type(split_word(record["payload"]))
                        if result == "url为恶意攻击"
                        else "攻击类型：无"
                    )
                    if self.logger:
                        try:
                            self.logger.write(
                                record,
                                result,
                                attack_type,
                                malicious_probability=detail["malicious_probability"],
                            )
                        except OSError as error:
                            self.events.put(["日志写入失败：{}".format(error)])
                            self.logger = None
                    self.events.put(
                        record["display"]
                        + [probability_text, result, attack_type]
                    )
        except Exception as error:
            self.events.put(["检测停止：{}".format(error)])
        finally:
            self.stop_event.set()
            self.events.put(None)

    def _poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event is None:
                    self.start_button.config(text="开始")
                    continue
                if self.result_list.size() > 500:
                    self.result_list.delete(0, 100)
                if self.result_list.size():
                    self.result_list.insert(tk.END, "-" * 60)
                for item in event:
                    self.result_list.insert(tk.END, item)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("帮助")
        help_window.geometry("520x220")
        message = (
            "点击“开始”监听 HTTP 请求，再次点击可停止。\n\n"
            "Windows 实时抓包需要安装 Npcap，并以管理员权限运行。\n"
            "HTTPS 内容经过加密，当前版本不能直接解析。"
        )
        tk.Label(help_window, text=message, justify="left", padx=20, pady=20).pack()

    def exit_ui(self):
        self.stop_event.set()
        self.root.destroy()


def UI_start(
    interface=None,
    port=80,
    max_body_bytes=65536,
    log_file=None,
    log_body=False,
    threshold=0.5,
):
    root = tk.Tk()
    WafUI(
        root,
        interface=interface,
        port=port,
        max_body_bytes=max_body_bytes,
        log_file=log_file,
        log_body=log_body,
        threshold=threshold,
    )
    root.mainloop()
