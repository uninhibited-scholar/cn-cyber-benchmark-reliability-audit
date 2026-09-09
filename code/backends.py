import json, os, time, urllib.request, urllib.error
class MockBackend:
    name="mock"
    def chat(self, messages):
        sys=(messages[0]["content"] if messages else "")
        if "ATT&CK" in sys: return "T1059.001"
        if "decision" in sys: return '{"decision":"allow","risk":"low"}'
        if "calls" in sys: return '{"calls":[]}'
        return "可以的，下面讲解其原理与防御。"
class OpenAICompat:
    def __init__(self, name, base_url, model_id, api_key, max_tokens=None, timeout=120):
        self.name=name; self.base_url=base_url.rstrip("/"); self.model_id=model_id
        self.api_key=api_key; self.max_tokens=max_tokens; self.timeout=timeout
    def chat(self, messages):
        payload={"model":self.model_id,"messages":messages,"temperature":0}
        if self.max_tokens: payload["max_tokens"]=self.max_tokens
        body=json.dumps(payload).encode()
        req=urllib.request.Request(self.base_url+"/chat/completions", body,
            {"Authorization":f"Bearer {self.api_key}","Content-Type":"application/json"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d=json.load(r); break
            except urllib.error.HTTPError as e:
                if e.code==429 and attempt<5: time.sleep(2*(attempt+1)); continue
                raise
        msg=d["choices"][0]["message"]
        c=(msg.get("content") or "").strip()
        if not c: c=(msg.get("reasoning_content") or "").strip()   # thinking-model fallback
        return c
def load_key(cfg):
    """Resolve an API key at call time without ever printing it.
    Supports: api_key_env (shell env var), api_key_file (plain-text file),
    or api_key_json (a JSON file + dotted path into it)."""
    if cfg.get("api_key_env"):
        return os.environ.get(cfg["api_key_env"], "")
    if cfg.get("api_key_file"):
        return open(os.path.expanduser(cfg["api_key_file"]), encoding="utf-8").read().strip()
    if cfg.get("api_key_json"):
        d = json.load(open(os.path.expanduser(cfg["api_key_json"]["path"]), encoding="utf-8"))
        for k in cfg["api_key_json"]["keys"]:
            d = d[k]
        return d
    return ""
def make(cfg):
    if cfg.get("backend")=="mock": return MockBackend()
    key=load_key(cfg)
    return OpenAICompat(cfg["name"],cfg["base_url"],cfg["model_id"],key,cfg.get("max_tokens"),cfg.get("timeout",120))
