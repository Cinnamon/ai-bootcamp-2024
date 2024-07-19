# rag-foundation-exercise

## Installation

**Note:** Prefer `python=3.10.*`

### 1. Clone repo (If you've already done this, jump to step 2)
```sh
git clone https://github.com/Cinnamon/ai-bootcamp-2024
cd ai-bootcamp-2024
```
### 2. Update changes from remote repo
Make sure you have folder `rag-foudation` in `ai-bootcamp-2024`

```sh
git checkout main
git fetch origin
```
### 3. Create a new environment for this week homework
#### Windows
   - **Open Command Prompt.**
   - **Navigate to your project directory:**
```sh
cd C:\Path\To\ai-bootcamp-2024
```
   - **Create a virtual environment using Python 3.10:**

Check your python version first using `py -0` or `where python`

```
python -m venv rag-foundation
or
path/to/python3.10 -m venv rag-foundation
```

   - **Activate the Virtual Environment:**
```sh
rag-foundation\Scripts\activate
```

#### Ubuntu/MacOS

- **Open a terminal.**
- **Create a new Conda environment with Python 3.10:**
```sh
conda create --name rag-foundation python=3.10
```

- **Activate the Conda Environment:**
```sh
conda activate rag-foundation
```

### 4. **Install Required Packages:**
   - Install the required packages from `requirements.txt`:
```sh
pip install -r requirements.txt
```

### 5. **Fill your implementation**

Search for `"Your code here"` line in the codebase which will lead you to where you should place your code.

### 6. **Run Script:**
You should read the code in this repository carefully to understand the setup comprehensively.

You can run the script below to get the results from your pre-built RAG, for example:
```sh
python -m scripts.main \
   --data_path <your_data_path> \
   --output_path predictions.jsonl \
   --mode <your_mode> \
   --force_index <True or False> \
   --retrieval_only True \
   --top_k 5
```
where some arguments can be:
   - `mode`: `sparse` or `semantic`
   - `force_index`: `True` or `False` (True: override the old vectorstore index)
   - `retrieval_only`: `True` or `False` (True: just get the retrieval contexts, answers are empty)

#### NOTE:
To use LLM generation with RAG pipeline, you can use ChatOpenAI by supplying OPENAI_API_KEY in the enviroment variable (supposed you have one).
If you don't have access to OpenAI API, use Groq free-tier instead:
- Register an account at https://console.groq.com/keys (free)
- Generate your API key
- Assign env variable: `export GROQ_API_KEY=<YOUR API KEY>`
- Run the main script without `--retrieval_only` to use LLM

### 7. **Run Evaluation:**
```sh
python evaluate.py --predictions predictions.jsonl --gold data/qasper-test-v0.3.json --retrieval_only True
```
where some arguments can be:
   - `retrieval_only`: `True` or `False` (True: just evaluate the retrieval contexts)
