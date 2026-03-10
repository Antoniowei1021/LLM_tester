from flask import Flask, render_template, request, redirect, url_for, jsonify
import requests
import time
import threading
app = Flask(__name__)
# This program is a LLM pressure tester. It will send a large number of requests to the LLM and 
# measure the response time and token/second. The results will be displayed in a web interface.
OLLAMA_CHAT_URL = "http://localhost:11434/v1/chat/completions"
cancel_flag = False
latest_test = None
run_id = 0
cancel_run_id = None
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

def run_single_request(payload: dict, idx: int, results: list):
    """One worker request to Ollama."""
    start = time.time()
    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=300, stream=True)
        latency = time.time() - start
        

        if resp.status_code != 200:
            results[idx] = {
                "ok": False,
                "status_code": resp.status_code,
                "latency": time.time() - start,
                "token_per_sec": 0,
                "completion_tokens": 0,
                "response": "",
                "error": resp.text,
            }
            return

        text_buffer = ""
        first_token_time = None

        for line in resp.iter_lines():
            if not line:
                continue

            decoded = line.decode("utf-8", errors="ignore").strip()

            if decoded.startswith("data:"):
                decoded = decoded[5:].strip()

            if decoded == "[DONE]":
                break

            try:
                j = requests.models.complexjson.loads(decoded)
            except Exception:
                continue

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
                if results[idx] is None:
                    results[idx] = {
                        "ok": False,
                        "status_code": 0,
                        "latency": 0,
                        "token_per_sec": 0,
                        "completion_tokens": 0,
                        "response": "",
                        "error": ""
                    }
                results[idx]["response"] = text_buffer
                results[idx]["latency"] = time.time() - start
                latest_test["runs"] = results
                # If this was a full non-stream JSON body, we already have the whole answer.
                if j.get("choices", [{}])[0].get("message"):
                    break

        end = time.time()

        latency = end - start
        generation = (end - first_token_time) if first_token_time else latency

        completion_tokens = len(text_buffer.split())
        tps = completion_tokens / generation if generation > 0 else 0

        if results[idx] is None:
            results[idx] = {
                "ok": False,
                "status_code": 0,
                "latency": 0,
                "token_per_sec": 0,
                "completion_tokens": 0,
                "response": "",
                "error": ""
            }
        results[idx].update({
            "ok": True,
            "status_code": 200,
            "latency": latency,
            "token_per_sec": tps,
            "completion_tokens": completion_tokens,
            "response": text_buffer,
            "error": "",
        }
)
    except Exception as e:
        latency = time.time() - start
        results[idx] = {
            "ok": False,
            "status_code": 0,
            "latency": time.time() - start,
            "token_per_sec": 0,
            "completion_tokens": 0,
            "response": "",
            "error": str(e),
        }

@app.route('/submitPrompt', methods=['POST'])
def submit():
    # add multi threading or multiprocessing here to send multiple requests to the LLM at the same time
    # and store the results in a list or a database, then return the results to the frontend to display
    global latest_test # call the LLM API with the prompt and get the response, response time and tokens per second
    global cancel_flag
    global run_id
    global cancel_run_id
    data = request.get_json(silent=True) or {}# get response from frontend and store it in data dictionary
    latest_test = data
    my_run_id = run_id
    run_id += 1
    latest_test["run_id"] = my_run_id
    latest_test["status"] = "running"
    try:
        num_users = int(latest_test.get("num_users", 1))
        payload = {
        "model": latest_test.get("model"),
        "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": latest_test.get("prompt")}
            ],
        "stream": True
        }
        mt = latest_test.get("max_tokens")
        if mt and str(mt).isdigit():
            payload["max_tokens"] = int(mt)
        threads = []
        results = [{
            "ok": False,
            "status_code": 0,
            "latency": 0,
            "token_per_sec": 0,
            "completion_tokens": 0,
            "response": "",
            "error": ""
        } for _ in range(num_users)]
        latest_test["runs"] = results
        for i in range(num_users):
            t = threading.Thread(
                target=run_single_request,
                args=(payload, i, results)
            )
            threads.append(t)
            t.start()

        def finalize_results():
            for t in threads:
                t.join()

            ok_runs = [r for r in results if r and r.get("ok")]
            avg_latency = sum(r["latency"] for r in ok_runs) / len(ok_runs) if ok_runs else 0
            avg_tps = sum(r["token_per_sec"] for r in ok_runs) / len(ok_runs) if ok_runs else 0

            latest_test["runs"] = results
            latest_test["latency_sec"] = round(avg_latency, 3)
            latest_test["token_per_sec"] = round(avg_tps, 3)
            first_resp = ""
            for r in ok_runs:
                if r.get("response"):
                    first_resp = r.get("response", "")
                    break
            latest_test["response_text"] = first_resp
            latest_test["status"] = "completed"
            latest_test["error"] = "None"

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

if __name__ == '__main__':    app.run(debug=True)
