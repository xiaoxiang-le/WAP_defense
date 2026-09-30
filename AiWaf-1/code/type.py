XSS_RULES = {
    "严重": {
        "<script",
        "<script>",
        "</script>",
        "<iframe",
        "<iframe>",
        "</iframe>",
        "response",
        "write(",
        "eval(",
        "prompt(",
        "alert(",
        "javascript:",
        "document",
        "cookie",
    },
    "中级": {
        "onclick=",
        "onerror=",
        "<base",
        "<base>",
        "</base>",
        "location",
        "hash",
        "window",
        "name",
        "<form",
        "<form>",
        "</form>",
    },
    "轻微": {"echo", "print", "href=", "sleep"},
}

SQL_RULES = {
    "严重": {
        "xp_",
        "substr",
        "utl_",
        "benchmark",
        "shutdown",
        "@@version",
        "information_schema",
        "hex(",
    },
    "中级": {
        "select",
        "if(",
        "union",
        "group",
        "by",
        "count(",
        "/**/",
        "char(",
        "drop",
        "delete",
        "concat",
        "orderby",
        "case",
        "when",
        "ascii(",
        "exec(",
        "length",
    },
    "轻微": {
        "and",
        "or",
        "like",
        "from",
        "insert",
        "update",
        "create",
        "else",
        "exists",
        "table",
        "database",
        "where",
        "mid",
        "updatexml(",
        "null",
        "sqlmap",
        "md5(",
        "floor",
        "rand",
        "cast",
        "dual",
        "fetch",
        "declare",
        "cursor",
        "extractvalue(",
        "join",
        "convert",
        "distinct",
    },
}


def _match_rules(tokens, rules):
    normalized = {str(token).lower() for token in tokens}
    for severity in ("严重", "中级", "轻微"):
        if normalized.intersection(rules[severity]):
            return severity
    return None


def find_type(url_tokens):
    xss_severity = _match_rules(url_tokens, XSS_RULES)
    if xss_severity:
        return "攻击类型：XSS  攻击等级：{}".format(xss_severity)

    sql_severity = _match_rules(url_tokens, SQL_RULES)
    if sql_severity:
        return "攻击类型：SQL注入  攻击等级：{}".format(sql_severity)

    return "攻击类型：未知  攻击等级：未知"
