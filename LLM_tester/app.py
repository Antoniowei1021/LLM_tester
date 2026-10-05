import statistics
from flask import Flask, render_template, request, jsonify  # type: ignore[import-not-found]
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
import requests
import time
import threading
import matplotlib
from io import BytesIO
import base64
matplotlib.use("Agg")
app = Flask(__name__)
# This program is a LLM pressure tester. It will send a large number of requests to the LLM and 
# measure the response time and token/second. The results will be displayed in a web interface.
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
latest_test = None
run_id = 0
history = []
cancel_events = {}
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
    with state_lock:
        if latest_test is None:
            return jsonify({
                "ok": False,
                "message": "No active test to cancel"
            }), 400

        current_run_id = latest_test.get("run_id")
        cancel_event = cancel_events.get(current_run_id)

        if cancel_event is None:
            return jsonify({
                "ok": False,
                "message": "No active test to cancel"
            }), 400
        try:
            cancel_event.set()
        except Exception as e:
            return jsonify({
                "ok": False,
                "message": f"No event to cancel: {str(e)}"
            }), 500

        latest_test["status"] = "cancelled"
        latest_test["error"] = "Test cancelled by user"

    return jsonify({
        "ok": True,
        "message": "Test cancelled",
        "run_id": current_run_id
    }), 200

def run_single_request(payload: dict, idx: int, results: list, test_start: float, cancel_event):
    """One worker request to Ollama using the native /api/chat endpoint."""
    start = time.time()
    with state_lock:
        results[idx]["status"] = "request_started"
        results[idx]["started_at_offset_sec"] = round(start - test_start, 3)

    if cancel_event.is_set():
        with state_lock:
            results[idx].update({
                "ok": False,
                "status": "cancelled",
                "latency": 0,
                "ttft_sec": None,
                "generation_sec": None,
                "token_per_sec": None,
                "completion_tokens": None,
                "response": "",
                "error": "Cancelled by user",
            })
        return

    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=300, stream=True)

        if resp.status_code != 200:
            with state_lock:
                results[idx] = {
                    "ok": False,
                    "status": "error",
                    "status_code": resp.status_code,
                    "latency": time.time() - start,
                    "ttft_sec": None,
                    "started_at_offset_sec": results[idx].get("started_at_offset_sec"),
                    "first_token_at_offset_sec": None,
                    "generation_sec": None,
                    "token_per_sec": None,
                    "completion_tokens": None,
                    "response": "",
                    "error": resp.text,
                }
            return

        text_buffer = ""
        first_token_time = None
        eval_count = None
        eval_duration_ns = None

        for line in resp.iter_lines():
            if cancel_event.is_set():
                end = time.time()
                resp.close()
                with state_lock:
                    results[idx].update({
                        "ok": False,
                        "status": "cancelled",
                        "status_code": 200,
                        "latency": end - start,
                        "ttft_sec": (first_token_time - start) if first_token_time is not None else None,
                        "first_token_at_offset_sec": (
                            round(first_token_time - test_start, 3)
                            if first_token_time is not None
                            else None
                        ),
                        # A cancelled stream never receives Ollama's final eval metrics.
                        "generation_sec": None,
                        "token_per_sec": None,
                        "completion_tokens": None,
                        "response": text_buffer,
                        "error": "Cancelled by user",
                    })
                return

            if not line:
                continue

            decoded = line.decode("utf-8", errors="ignore").strip()
            if not decoded:
                continue

            try:
                j = requests.models.complexjson.loads(decoded)
            except Exception:
                continue

            message = j.get("message") or {}
            thinking_delta = message.get("thinking") or ""
            content_delta = message.get("content") or ""
            delta = thinking_delta + content_delta

            if delta:
                now = time.time()
                if first_token_time is None:
                    first_token_time = now
                text_buffer += delta

                with state_lock:
                    results[idx]["response"] = text_buffer
                    results[idx]["latency"] = now - start
                    results[idx]["ttft_sec"] = first_token_time - start
                    results[idx]["status"] = "streaming"
                    results[idx]["first_token_at_offset_sec"] = round(
                        first_token_time - test_start, 3
                    )
                    # Native eval metrics are only authoritative when Ollama sends done=true.
                    results[idx]["generation_sec"] = None
                    results[idx]["completion_tokens"] = None
                    results[idx]["token_per_sec"] = None

            if j.get("done"):
                eval_count = j.get("eval_count")
                eval_duration_ns = j.get("eval_duration")
                break

        end = time.time()
        latency = end - start
        ttft = (first_token_time - start) if first_token_time is not None else None

        generation_sec = None
        tps = None
        if isinstance(eval_duration_ns, (int, float)) and eval_duration_ns > 0:
            generation_sec = eval_duration_ns / 1_000_000_000
            if isinstance(eval_count, (int, float)):
                tps = eval_count / generation_sec

        with state_lock:
            results[idx].update({
                "ok": True,
                "status": "completed",
                "status_code": 200,
                "latency": latency,
                "ttft_sec": ttft,
                "first_token_at_offset_sec": (
                    round(first_token_time - test_start, 3)
                    if first_token_time is not None
                    else None
                ),
                "generation_sec": generation_sec,
                "token_per_sec": tps,
                "completion_tokens": eval_count,
                "response": text_buffer,
                "error": "",
            })

    except Exception as e:
        with state_lock:
            results[idx] = {
                "ok": False,
                "status": "error",
                "status_code": 0,
                "latency": time.time() - start,
                "ttft_sec": None,
                "first_token_at_offset_sec": None,
                "generation_sec": None,
                "token_per_sec": None,
                "completion_tokens": None,
                "response": "",
                "error": str(e),
            }

@app.route('/submitPrompt', methods=['POST'])
def submit():
    # using global variables to store the latest test configuration and results, as well as the cancel flag and run id
    global latest_test # latest_test cannot be replaced by new test case.
    global run_id
    test_state = request.get_json(silent=True) or {}# create a local variable "test_state" for the current test case, then add to latest_test
    num_users = int(test_state.get("num_users", 1)) # get the number of users from the latest_test dictionary, default to 1 if not provided
    results = [{ # normalize the results list to have a dictionary for each user, with default values for the metrics
                "ok": False,
                "status": "pending",
                "status_code": 0,
                "latency": 0,
                "ttft_sec": None,
                "started_at_offset_sec": None,
                "first_token_at_offset_sec": None,
                "generation_sec": None,
                "token_per_sec": None,
                "completion_tokens": None,
                "response": "",
                "error": ""
                } for _ in range(num_users)]
    cancel_event = threading.Event() # create a threading event to signal cancellation to the worker threads
    with state_lock: 
        my_run_id = run_id 
        run_id += 1
        cancel_events[my_run_id] = cancel_event
        test_state["run_id"] = my_run_id
        test_state["status"] = "running"
        test_state["runs"] = results
        test_state["ttft_sec"] = 0
        test_state["total_latency_sec"] = 0
        test_state["throughput_tps"] = 0
        test_state["total_tokens"] = 0
        test_state["request_start_spread_sec"] = 0
        test_state["first_token_spread_sec"] = 0
        test_state["queue_like_behavior"] = False
        latest_test = test_state
    try:
        test_start = time.time() # start the test and record the start time, this will be used to calculate the latency and throughput of the test
        with state_lock:
            test_state["test_start_time"] = test_start
# 删除原第 318 行，num_users 已经在第 285 行读取过了
            payload = {
                "model": test_state.get("model"),
                "messages": [
                    {"role": "user", "content": test_state.get("prompt")}
                ],
                "stream": True,
                "options": {},
            }
            mt = test_state.get("max_tokens")
        if mt and str(mt).isdigit():
            payload["options"]["num_predict"] = int(mt)
        threads = []
        for i in range(num_users):
            request_payload = dict(payload)
            request_payload["messages"] = [dict(m) for m in payload["messages"]]
            t = threading.Thread(target=run_single_request,args=(request_payload, i, results, test_start, cancel_event), daemon=True)
            threads.append(t)
        for t in threads:
            t.start()
        def finalize_results():
            # Poll all worker threads together so the frontend can see every request updating in real time.
            while any(t.is_alive() for t in threads):
                time.sleep(0.1)
            for t in threads:
                t.join()
            if cancel_event.is_set():
                with state_lock:
                    test_state["status"] = "cancelled"
                    test_state["error"] = "Test cancelled by user"
                    cancel_events.pop(my_run_id, None)
                return
            # after all threads are done, we can calculate the average latency and tokens per second, and update the latest_test with the final results to trigger the frontend update 
            ok_runs = [r for r in results if r and r.get("ok")]
            avg_latency = sum(r["latency"] for r in ok_runs) / len(ok_runs) if ok_runs else 0
            tps_runs = [r["token_per_sec"] for r in ok_runs if r.get("token_per_sec") is not None]
            avg_tps = statistics.mean(tps_runs) if tps_runs else 0

            ttft_runs = [r["ttft_sec"] for r in ok_runs if r.get("ttft_sec") is not None]
            avg_ttft = statistics.mean(ttft_runs) if ttft_runs else 0

            token_runs = [r["completion_tokens"] for r in ok_runs if r.get("completion_tokens") is not None]
            total_tokens = sum(token_runs)
            total_time = max((r["latency"] for r in ok_runs), default=0)
            throughput_tps = (total_tokens / total_time) if total_time > 0 else 0

            started_offsets = [
                r.get("started_at_offset_sec")
                for r in ok_runs
                if r.get("started_at_offset_sec") is not None
            ]
            first_token_offsets = [
                r.get("first_token_at_offset_sec")
                for r in ok_runs
                if r.get("first_token_at_offset_sec") is not None
            ]
            request_start_spread = (
                max(started_offsets) - min(started_offsets)
                if len(started_offsets) >= 2
                else 0
            )
            first_token_spread = (
                max(first_token_offsets) - min(first_token_offsets)
                if len(first_token_offsets) >= 2
                else 0
            )
            # If all HTTP requests started almost together but first tokens arrived far apart,
            # the bottleneck is likely the Ollama/model scheduler queue, not Flask threading.
            queue_like_behavior = request_start_spread < 0.5 and first_token_spread > 1.0

            with state_lock:
                test_state["latency_sec"] = round(avg_latency, 3)
                test_state["ttft_sec"] = round(avg_ttft, 3)
                test_state["token_per_sec"] = round(avg_tps, 3)
                test_state["throughput_tps"] = round(throughput_tps, 3)
                test_state["total_tokens"] = total_tokens
                test_state["total_latency_sec"] = round(total_time, 3)
                test_state["request_start_spread_sec"] = round(request_start_spread, 3)
                test_state["first_token_spread_sec"] = round(first_token_spread, 3)
                test_state["queue_like_behavior"] = queue_like_behavior
                test_state["model"] = payload["model"]
                first_resp = ""
                for r in ok_runs:
                    if r.get("response"):
                        first_resp = r.get("response", "")
                        break
                test_state["response_text"] = first_resp
                test_state["status"] = "completed"
                test_state["error"] = "None"
                history.append({
                    "model": payload["model"],
                    "num_users": num_users,
                    "ttft_sec": round(avg_ttft, 2),
                    "token_per_sec": round(avg_tps, 2),
                    "throughput_tps": round(throughput_tps, 2),
                    "request_start_spread_sec": round(request_start_spread, 2),
                    "first_token_spread_sec": round(first_token_spread, 2),
                    "queue_like_behavior": queue_like_behavior
                })
                cancel_events.pop(my_run_id, None)
        threading.Thread(target=finalize_results, daemon=True).start()

    except Exception as e:
        test_state["status"] = "error" # if error
        test_state["error"] = str(e)
        return jsonify({"message": "Error calling LLM API", "error": str(e)}), 500
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
        ax.set_title('Performance vs Concurrency (Ollama Native Metrics)')
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
     