This is a rudimentary pressure tester for personal LLMs. This tool is based on Ollama and Python Flask. It can automatically detect models installed on users' PCs and run tests. It can test prompts, model responses, latencies, token/s, and multi-thread capabilities. 

## Requirements

- Python >= 3.9.6
- Ollama installed
- pip

## Setup

### 1. Download and Install Ollama

First, in ollama.com, use this link:
```
https://ollama.com/download
```
Or you can use your terminal and run the following:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Start the service

After installation, start the Ollama service:

```bash
ollama serve
```
Then pull the models you want to test, for example:

```bash
ollama pull deepseek-r1:1.5b
```
This is just for reference. You can pull multiple models within the PC's capacity. 
For people whose RAM <= 16GB, models larger than 16b are not recommended. 

You can see the models you pulled with this command:
```bash
ollama list
```

### 3. Python dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate
```

Install:

pip install -r requirements.txt


### 4. Run the service

First, navigate to your folder path
Then in the terminal, run:
```
python app.py
```
This web server will run on http://localhost:5000
Click on the link, and you shhould be good to go.

### 5. Usage 
1.	Open the web interface.
2.	Configure the parameters:

The default api is this:
```
http://localhost:11434/v1
```

  •	Model
Automatically detected from your local Ollama installation.
	•	Concurrent Users
Number of simultaneous LLM requests.
	•	Prompt
The prompt used for testing.

3. Start the test
Each concurrent request will appear in a separate result window showing:
	•	latency
	•	tokens generated
	•	tokens per second
	•	streamed model 

```
Browser
   │
   │  HTTP
   ▼
Flask Backend
   │
   │  OpenAI-compatible API
   ▼
Ollama (localhost:11434)
   │
   ▼
Local LLM Models
```

## Features
	•	Concurrent LLM request testing
	•	Streaming token output
	•	Real-time performance metrics
	•	Automatic detection of local Ollama models
	•	Simple web interface
	•	Compatible with OpenAI-style API


## Notes
	•	Ollama must be running before starting the server.
	•	Large models may significantly affect latency and token throughput.
	•	Concurrent tests depend on your machine’s CPU/GPU capacity.


