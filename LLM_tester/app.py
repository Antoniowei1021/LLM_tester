import statistics
from flask import Flask, render_template, request, jsonify
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
import requests
import time
import threading
import matplotlib
from io import BytesIO
import base64
import re
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

matplotlib.use("Agg")
app = Flask(__name__)
# This program is a LLM pressure tester. It will send a large number of requests to the LLM and 
# measure the response time and token/second. The results will be displayed in a web interface.
OLLAMA_CHAT_URL = "http://localhost:11434/v1/chat/completions"
cancel_flag = False
latest_test = None
run_id = 0
cancel_run_id = None
history = []
state_lock = threading.Lock()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET'])
def get_result():
    # retrive the results of the latest test and return it as json
    if latest_test is None:
        return jsonify({"message": "No test results available"})
    return jsonify(latest_test), 200

@app.route('/halt', methods=['POST'])
def halt():
    # stop the LLM
    global latest_test
    global cancel_flag
    global cancel_run_id
    cancel_flag = True
    if latest_test is not None:
        cancel_run_id = latest_test.get("run_id")
        latest_test["status"] = "cancelled"
    else: 
        latest_test = {"status": "cancelled", "error": "No active test to cancel"}
    return jsonify({"message": "System halted"})

def estimate_token_count(text: str) -> int:
    """Estimate token count cross-language fallback."""
    if not text:
        return 0
    pattern = re.compile(r'\w+|[^\w\s]', re.UNICODE)
    return len(pattern.findall(text))


TOKENIZER_CACHE = {}


def resolve_tokenizer_name(model_name: str) -> str:
    """
    Map Ollama/local model names to a Hugging Face tokenizer name.
    This is heuristic but good enough for benchmark consistency.
    """
    if not model_name:
        return "bert-base-uncased"

    name = model_name.lower()

    if "qwen" in name:
        # Qwen family
        if "2.5" in name:
            return "Qwen/Qwen2.5-1.5B-Instruct"
        return "Qwen/Qwen2-1.5B-Instruct"

    if "llama" in name or "llama3" in name or "llama-3" in name:
        return "meta-llama/Meta-Llama-3-8B-Instruct"

    if "deepseek" in name:
        return "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if "mistral" in name:
        return "mistralai/Mistral-7B-Instruct-v0.3"

    if "gemma" in name:
        return "google/gemma-2-2b-it"

    # Fallback tokenizer for unknown models
    return "bert-base-uncased"


def get_tokenizer(model_name: str):
    """Load and cache a tokenizer for the selected model family."""
    if AutoTokenizer is None:
        return None

    tok_name = resolve_tokenizer_name(model_name)
    if tok_name in TOKENIZER_CACHE:
        return TOKENIZER_CACHE[tok_name]

    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
        TOKENIZER_CACHE[tok_name] = tokenizer
        return tokenizer
    except Exception:
        return None


def count_tokens_for_model(text: str, model_name: str) -> int:
    """
    Prefer a model-family tokenizer; fall back to cross-language estimate.
    """
    if not text:
        return 0

    tokenizer = get_tokenizer(model_name)
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass

    return estimate_token_count(text)

def run_single_request(payload: dict, idx: int, results: list, test_start: float):
    """One worker request to Ollama."""
    start = time.time()
    with state_lock:
        if results[idx] is not None:
            results[idx]["status"] = "request_started"
            results[idx]["started_at_offset_sec"] = round(start - test_start, 3)
            if latest_test is not None:
                latest_test["runs"] = results
    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=300, stream=True)
        latency = time.time() - start
        

        if resp.status_code != 200: # when post request fails, record the error and return
            with state_lock:
                results[idx] = {
                    "ok": False,
                    "status": "error",
                    "status_code": resp.status_code,
                    "latency": time.time() - start,
                    "ttft_sec": None,
                    "first_token_at_offset_sec": None,
                    "generation_sec": 0,
                    "token_per_sec": 0,
                    "completion_tokens": 0,
                    "response": "",
                    "error": resp.text,
                }
                if latest_test is not None:
                    latest_test["runs"] = results
            return
        # For streaming responses, we need to iterate through the lines and concatenate the text until we get the full response.
        # We also want to record the time to first token (TTFT) and the token generation speed. For non-streaming responses,
        # we can just read the full response and calculate the metrics.
        text_buffer = ""
        first_token_time = None

        for line in resp.iter_lines(): # iterate through the streaming response line by line
            if not line:
                continue

            decoded = line.decode("utf-8", errors="ignore").strip() # decode the line and strip it, but ignore any decoding errors
            # the streaming response from Ollama is in the format of "data: {json}", so we need to remove the "data: " prefix and parse the json
            #but this part is still working with the json, not the multithreading part
            if decoded.startswith("data:"): 
                decoded = decoded[5:].strip()

            if decoded == "[DONE]":
                break

            try: # try to parse the json, but if it fails, just ignore the error and continue, because some lines might not be valid json (like the "data: " prefix or empty lines)
                j = requests.models.complexjson.loads(decoded)
            except Exception:
                continue
            # delta is the new text generated by the LLM in this line, but if it's not a streaming response, we can also get the full response from the json body. So we need to check both cases. This part is still working with the json, not the multithreading part.
            delta = (
                j.get("choices", [{}])[0]
                .get("delta", {})
                .get("content", "")
            )
            # Fallback for non-stream/full JSON responses
            if not delta:
                delta = (
                    j.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

            if delta:
                if first_token_time is None:
                    first_token_time = time.time()

                text_buffer += delta

                with state_lock:
                    if results[idx] is None:
                        results[idx] = {
                            "ok": False,
                            "status_code": 0,
                            "latency": 0,
                            "ttft_sec": None,
                            "generation_sec": 0,
                            "token_per_sec": 0,
                            "completion_tokens": 0,
                            "response": "",
                            "error": ""
                        }
                    results[idx]["response"] = text_buffer
                    results[idx]["latency"] = time.time() - start
                    results[idx]["ttft_sec"] = first_token_time - start if first_token_time else None
                    results[idx]["status"] = "streaming"
                    results[idx]["first_token_at_offset_sec"] = round(first_token_time - test_start, 3) if first_token_time else None
                    if latest_test is not None:
                        latest_test["runs"] = results
                # If this was a full non-stream JSON body, we already have the whole answer.
                if j.get("choices", [{}])[0].get("message"):
                    break

        end = time.time()

        latency = end - start
        ttft = (first_token_time - start) if first_token_time else latency
        generation = (end - first_token_time) if first_token_time else latency
        # if chinese, use chinese character splitter to count tokens
        # else, use split for latin languages
        completion_tokens = count_tokens_for_model(text_buffer, payload.get("model", ""))
        tps = completion_tokens / generation if generation > 0 else 0
        if results[idx] is None: 
            # idx is the index of the current request in the results list, which corresponds to the user number (0 to num_users-1)
            # if results[idx] is None, it means the request failed before we could record any response, so we should record it as a 
            # failure with the error message. This can happen if the request fails immediately or if the streaming response never yields any valid lines.
            results[idx] = {
                "ok": False,
                "status": "empty_response",
                "status_code": 0,
                "latency": 0,
                "ttft_sec": None,
                "started_at_offset_sec": None,
                "first_token_at_offset_sec": None,
                "generation_sec": 0,
                "token_per_sec": 0,
                "completion_tokens": 0,
                "response": "",
                "error": ""
            }
        with state_lock:
            results[idx].update({
                "ok": True,
                "status": "completed",
                "status_code": 200,
                "latency": latency,
                "ttft_sec": ttft,
                "first_token_at_offset_sec": round(first_token_time - test_start, 3) if first_token_time else None,
                "generation_sec": generation,
                "token_per_sec": tps,
                "completion_tokens": completion_tokens,
                "response": text_buffer,
                "error": "",
            })
            if latest_test is not None:
                latest_test["runs"] = results
    except Exception as e: # if the request fails, record the error and return
        latency = time.time() - start
        with state_lock:
            results[idx] = {
                "ok": False,
                "status": "error",
                "status_code": 0,
                "latency": time.time() - start,
                "ttft_sec": None,
                "first_token_at_offset_sec": None,
                "generation_sec": 0,
                "token_per_sec": 0,
                "completion_tokens": 0,
                "response": "",
                "error": str(e),
            }
            if latest_test is not None:
                latest_test["runs"] = results

@app.route('/submitPrompt', methods=['POST'])
def submit():
    # add multi threading or multiprocessing here to send multiple requests to the LLM at the same time
    # and store the results in a list or a database, then return the results to the frontend to display
    global latest_test # call the LLM API with the prompt and get the response, response time and tokens per second
    global cancel_flag
    global run_id
    global cancel_run_id
    global history
    data = request.get_json(silent=True) or {}# get response from frontend and store it in data dictionary
    latest_test = data
    my_run_id = run_id
    run_id += 1
    latest_test["run_id"] = my_run_id
    latest_test["status"] = "running"
    latest_test["ttft_sec"] = 0
    latest_test["total_latency_sec"] = 0
    latest_test["throughput_tps"] = 0
    latest_test["total_tokens"] = 0
    try:
        test_start = time.time()
        latest_test["test_start_time"] = test_start
        num_users = int(latest_test.get("num_users", 1))
        payload = {
            "model": latest_test.get("model"),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": latest_test.get("prompt")}
            ],
            "stream": True,
            "options": {}
        }
        mt = latest_test.get("max_tokens")
        if mt and str(mt).isdigit():
            payload["max_tokens"] = int(mt)
            payload["options"]["num_predict"] = int(mt)
        threads = []
        results = [{
    "ok": False,
    "status": "pending",
    "status_code": 0,
    "latency": 0,
    "ttft_sec": None,
    "started_at_offset_sec": None,
    "first_token_at_offset_sec": None,
    "generation_sec": 0,
    "token_per_sec": 0,
    "completion_tokens": 0,
    "response": "",
    "error": ""
} for _ in range(num_users)]
        with state_lock:
            latest_test["runs"] = results
        for i in range(num_users): # start num_users threads to send requests to the LLM at the same time, and store the results in the results list
            request_payload = dict(payload)
            request_payload["messages"] = [dict(m) for m in payload["messages"]]
            t = threading.Thread(target=run_single_request, args=(request_payload, i, results, test_start), daemon=True)
            threads.append(t)
            t.start()
        def finalize_results():
            # Poll all worker threads together so the frontend can see every request updating in real time.
            while any(t.is_alive() for t in threads):
                with state_lock:
                    latest_test["runs"] = results
                time.sleep(0.1)
            for t in threads:
                t.join()
            # after all threads are done, we can calculate the average latency and tokens per second, and update the latest_test with the final results to trigger the frontend update 
            ok_runs = [r for r in results if r and r.get("ok")]
            avg_latency = sum(r["latency"] for r in ok_runs) / len(ok_runs) if ok_runs else 0
            avg_tps = sum(r["token_per_sec"] for r in ok_runs) / len(ok_runs) if ok_runs else 0

            ttft_runs = [r["ttft_sec"] for r in ok_runs if r.get("ttft_sec") is not None]
            avg_ttft = statistics.mean(ttft_runs) if ttft_runs else 0

            total_tokens = sum(r["completion_tokens"] for r in ok_runs)
            total_time = max((r["latency"] for r in ok_runs), default=0)
            throughput_tps = (total_tokens / total_time) if total_time > 0 else 0

            with state_lock:
                latest_test["runs"] = results
                latest_test["latency_sec"] = round(avg_latency, 3)
                latest_test["ttft_sec"] = round(avg_ttft, 3)
                latest_test["token_per_sec"] = round(avg_tps, 3)
                latest_test["throughput_tps"] = round(throughput_tps, 3)
                latest_test["total_tokens"] = total_tokens
                latest_test["total_latency_sec"] = round(total_time, 3)
                latest_test["model"] = payload["model"]
                first_resp = ""
                for r in ok_runs:
                    if r.get("response"):
                        first_resp = r.get("response", "")
                        break
                latest_test["response_text"] = first_resp
                latest_test["status"] = "completed"
                latest_test["error"] = "None"
                history.append({
                    "model": payload["model"],
                    "num_users": num_users,
                    "ttft_sec": round(avg_ttft, 2),
                    "token_per_sec": round(avg_tps, 2),
                    "throughput_tps": round(throughput_tps, 2)
                })
        threading.Thread(target=finalize_results, daemon=True).start()

    except Exception as e:
        latest_test["status"] = "error"
        latest_test["error"] = str(e)
        return jsonify({"message": "Error calling LLM API", "error": str(e)}), 500
    if cancel_flag and cancel_run_id == my_run_id:
        latest_test["status"] = "cancelled" 
        latest_test["error"] = "Test cancelled by user"
        latest_test["response_text"] = ""
        latest_test["token_per_sec"] = 0
        cancel_flag = False
        return "Cancelled by user", 200
    return jsonify({"ok": True, "message": "running", "run_id": my_run_id}), 200

@app.route('/models', methods=['GET'])
def get_models():
    try:
        resp = requests.get("http://localhost:11434/v1/models", timeout=10)
        if resp.status_code != 200:
            return jsonify({"message": "Error fetching models", "error": resp.text}), 500
        j = resp.json()
        models = [m["id"] for m in j["data"]]
        return jsonify({"ok": True, "models": models}), 200
    except Exception as e:
        return jsonify({"message": "Error fetching models", "error": str(e)}), 500
    
@app.route('/stat', methods=['GET'])
def get_stats():
    try:
        if not history:
            return jsonify({"ok": False, "error": "No data"}), 400
        # sort by num_users
        # group by model, then sort each model by num_users
        grouped = {}
        for h in history:
            model = h.get("model", "unknown")
            grouped.setdefault(model, []).append(h)
        fig = Figure(figsize=(9, 5.2), dpi=120)
        FigureCanvas(fig)
        ax = fig.add_subplot(111)

        for model, items in grouped.items():
            items = sorted(items, key=lambda x: x["num_users"])
            x = [h["num_users"] for h in items]
            y_ttft = [h["ttft_sec"] for h in items]
            y_tps = [h["token_per_sec"] for h in items]

            # same color for same model:
            line = ax.plot(x, y_ttft, marker='o', label=f'{model} - TTFT')[0]
            color = line.get_color()
            ax.plot(x, y_tps, marker='o', linestyle='--', color=color, label=f'{model} - Token/s')

        ax.set_xlabel('Concurrency (num_users)')
        ax.set_ylabel('Value')
        ax.set_title('Performance vs Concurrency (Model-Aware Token Counting)')
        ax.legend(fontsize=8)
        ax.grid(True)
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return jsonify({
            "ok": True,
            "image": img_base64,
            "data": history
        }), 200
    except Exception as e:
        return jsonify({"message": "Error fetching statistics", "error": str(e)}), 500

@app.route('/delete_history', methods=['POST'])
def delete_history():
    global history, latest_test
    history = []
    latest_test = None
    return jsonify({"ok": True, "message": "History and results deleted"}), 200

if __name__ == '__main__':
     app.run(debug=True, use_reloader=False, threaded=True)
     