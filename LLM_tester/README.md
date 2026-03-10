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














## Python dependencies

Install:

pip install -r requirements.txt
