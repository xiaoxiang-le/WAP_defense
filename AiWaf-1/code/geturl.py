from urllib.parse import unquote

from scapy.all import Ether, IP, Raw, TCP, bind_layers, sniff
from scapy.sessions import TCPSession

try:
    import scapy_http.http as http
except ImportError:
    from scapy.layers import http


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def packet_to_record(packet, max_body_bytes=65536):
    if not packet.haslayer(TCP) or not packet.haslayer(http.HTTPRequest):
        return None

    request = packet[http.HTTPRequest]
    host = _decode(getattr(request, "Host", b""))
    path = _decode(getattr(request, "Path", b""))
    url = unquote(host + path)
    if not url:
        return None

    source_ip = _decode(packet[IP].src) if packet.haslayer(IP) else ""
    target_ip = _decode(packet[IP].dst) if packet.haslayer(IP) else ""
    source_mac = _decode(packet[Ether].src) if packet.haslayer(Ether) else ""
    target_mac = _decode(packet[Ether].dst) if packet.haslayer(Ether) else ""
    method = _decode(getattr(request, "Method", b""))
    user_agent = _decode(getattr(request, "User_Agent", b""))
    content_type = _decode(getattr(request, "Content_Type", b""))
    body_bytes = bytes(packet[Raw].load)[:max_body_bytes] if packet.haslayer(Raw) else b""
    body = _decode(body_bytes)
    detection_payload = url if not body else "{}\n{}".format(url, body)

    return {
        "url": url,
        "body": body,
        "payload": detection_payload,
        "display": [
            "IP_Src：" + source_ip,
            "IP_Dst：" + target_ip,
            "Mac_Src：" + source_mac,
            "Mac_Dst：" + target_mac,
            "URL：" + url,
            "Method：" + method,
            "User-Agent：" + user_agent,
            "Host：" + host,
            "Path：" + path,
            "Content-Type：" + content_type,
            "Body：" + body,
        ],
    }


def sniff_requests(interface=None, port=80, timeout=1, count=1, max_body_bytes=65536):
    records = []
    bind_layers(TCP, http.HTTP, sport=port)
    bind_layers(TCP, http.HTTP, dport=port)

    def callback(packet):
        record = packet_to_record(packet, max_body_bytes=max_body_bytes)
        if record:
            records.append(record)

    options = {
        "filter": "tcp port {}".format(port),
        "lfilter": lambda packet: packet.haslayer(http.HTTPRequest),
        "prn": callback,
        "count": count,
        "timeout": timeout,
        "store": False,
        "session": TCPSession,
    }
    if interface:
        options["iface"] = interface
    sniff(**options)
    return records
