# **Customer 360 Intelligence Copilot**
### End-to-end customer analytics and AI support project
### **Project overview**
This project demonstrates an end-to-end workflow for a synthetic grocery
quick-commerce business. It covers local data generation, data-quality
checks, cleaning and validation, SQLite storage, customer analytics,
lapse prediction, support-ticket triage, policy retrieval, and a
simulated refund-agent workflow.

All data and policy documents in this project are fictional.
Language-model interactions are handled locally through mock_llm.py;
no API key or live model service is required.
### **Project objectives**
- Build a repeatable data-ingestion and cleaning workflow.
- Store trusted datasets as cleaned CSV files and in SQLite.
- Analyse customer behaviour using Recency, Frequency and Monetary
  (RFM) measures.
- Segment customers with K-means and inspect clusters using
  diagnostics and PCA.
- Build a customer-lapse prediction model while avoiding target-period
  leakage.
- Triage support tickets with input guardrails, PII redaction, retry
  handling, category validation and human review.
- Retrieve policy passages with TF-IDF and cosine similarity.
- Demonstrate controlled refund tools, safe stopping, checkpointing
  and resume.
### **Workflow**
1. Generate synthetic source data.
2. Load and inspect the CSV exports.
3. Audit, clean and validate the data.
4. Save cleaned datasets and create the SQLite database.
5. Query the database with SQL.
6. Perform RFM analysis and customer segmentation.
7. Train and evaluate the lapse prediction model.
8. Run privacy-aware support-ticket triage.
9. Build and evaluate policy retrieval (RAG).
10. Run the simulated refund-agent scenarios and inspect the ledger and
    checkpoints.
### **Repository layout**
The notebook expects to be run from the project repository root. The
main files and folders used by the workflow are:
project-root/
├── Customer_360_Intelligence_Copilot.ipynb
├── generate_data.py
├── requirements.txt
├── mock_llm.py
├── src/
│   ├── data_pipeline.py
│   ├── customer_models.py
│   ├── triage.py
│   ├── policy_rag.py
│   └── refund_agent.py
├── data/
│   ├── raw/          # Generated source CSVs, scenarios and policy files
│   ├── clean/        # Cleaned datasets
│   └── instamart.db  # SQLite database created by the pipeline
├── outputs/          # Evaluation results, model files, ledgers and reports
└── state/            # Agent checkpoint JSON files
Some generated files and folders are created only after running the
notebook.
### **Requirements**
- Python installed on your computer.
- The packages listed in requirements.txt.
- VS Code with the Python and Jupyter extensions, or another Jupyter
  environment.
Use a virtual environment to keep project dependencies isolated.

**Create an isolated environment:**

   ```bash
   python -m venv .venv
   ```

**Activate it:**

   ```bash
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```

**Install the pinned project dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

**Regenerate the synthetic raw data and policy documents:**

   ```bash
   python generate_data.py
   ```
In VS Code, select the .venv interpreter using Python: Select
Interpreter. For notebooks, select the same environment as the
notebook kernel.

### **Run the notebook**

1. Open Customer_360_Intelligence_Copilot.ipynb.
2. Ensure the terminal and notebook use the project root and the
   .venv environment.
3. Run the notebook cells in order, from the setup and data-generation
   sections through the final evaluation and reproducibility sections.
4. Review the generated files under data/, outputs/ and state/.
The notebook sets up the project root and imports modules from src/.
Running from the repository root helps its relative paths resolve
correctly   

### **Project components**
#### **Part A --- Data preparation and SQL**
- Generate reproducible synthetic source exports using
  generate_data.py.
- Inspect dimensions, sample rows, data types, missing values and
  blank values.
- Audit identifiers, duplicates and invalid values.
- Standardise selected categories and handle invalid or missing values
  using the documented cleaning rules.
- Retain the latest order record for each order ID after removing
  exact duplicates.
- Save cleaned CSV files and persist tables in data/instamart.db.
- Query the database with SQLite and compare selected SQL totals with
  pandas calculations.
#### **Part B --- Customer analytics and lapse prediction**
- Calculate RFM measures for completed orders.
- Scale RFM features before K-means clustering.
- Compare candidate cluster counts using diagnostics and inspect PCA
  explained variance.
- Build a customer-lapse classification model.
- Use orders before 2025-09-01 for model features and reserve
  September for defining the target label.
- Fit preprocessing within the training pipeline to reduce data
  leakage.
- Interpret evaluation with metrics such as precision, recall, F1 and
  the confusion matrix, rather than relying on accuracy alone.
#### **Part C --- Support-ticket triage**
  **The triage workflow demonstrates:**
- Guardrails for empty and overlong messages.
- PII redaction (email, phone and order ID in the documented
  sequence).
- Retry handling for simulated transient errors.
- Validation of predicted categories against an allowed set.
- Keyword-based fallback and human review for uncertain cases.
- Token-budget monitoring.
Only redacted ticket text is passed as the user message to the local
mock classifier.
#### **Part D --- Policy RAG and refund agent**
  **Policy retrieval (RAG)**
- Split policy documents into chunks.
- Represent chunks with TF-IDF.
- Rank retrieved passages using cosine similarity.
- Ground answers in retrieved policy context and use a fallback when
  the context does not support an answer.
- Evaluate expected source/fact retrieval and unknown-question
  fallback behaviour.
#### **Refund-agent simulation**

The documented tool workflow looks up an order, retrieves policy
information, calculates an eligible refund, and either issues a
simulated refund or escalates the case for human review. The tool map
connects tool names to Python functions.

The notebook also demonstrates scenario evaluation, safe stopping
(timeout, token limit and maximum iterations), checkpoint creation, and
resuming after a simulated crash. The refund ledger is a local JSONL
simulation: no real payment or money movement takes place.
### **Generated artifacts**
Depending on which notebook cells have been run, generated artifacts
include:
- data/raw/ --- synthetic source CSVs, policy documents and agent
  scenarios.
- data/clean/ --- cleaned CSV datasets.
- data/instamart.db --- SQLite database.
- outputs/ --- customer segments, lapse model, triage results,
  policy evaluation results, agent runs, refund ledger and summary
  files.
- state/ --- JSON checkpoint files used by the refund-agent
  workflow.
JSONL files contain one JSON record per line. CSV files can be opened
with spreadsheet software or read with pandas.
Reproducibility
The synthetic data generator uses a fixed random seed so the source
exports can be recreated. To test a clean rebuild:
1. Keep the source code, notebook, requirements file and project
   configuration.
2. Remove generated data/output artifacts only if you no longer need
   them (for example, data/raw/, data/clean/, data/instamart.db,
   outputs/ and state/).
3. Re-run the setup, data-generation and notebook cells in order.
4. Compare the newly generated outputs with the project checks.
Do not delete files that contain work you need to preserve. Expected
evaluation values should be checked against generated results, not
hardcoded into the pipeline.
### **Important notes and limitations**
- The dataset and policy content are synthetic and intended for
  learning and demonstration.
- mock_llm.py provides local mock interactions; this project does
  not require an API key.
- The refund workflow is a simulation and does not move money.
- PII redaction is a defence-in-depth measure, not a guarantee that
  all sensitive information will be removed.
- Human review is appropriate for ambiguous, invalid or sensitive
  cases.
- The lapse-model retention list described in the notebook is
  illustrative where its segment flag uses information from the full
  observation window.
### **Troubleshooting**
**ModuleNotFoundError: No module named 'src'** - Open the project
folder (not just the notebook file) in VS Code. - Run the notebook from
the repository root. - Confirm the setup cell adds the repository root
to sys.path.

**PowerShell says running scripts is disabled** - Use Command Prompt and
run .venv\Scripts\activate.bat, or follow the PowerShell activation
instructions above if your device policy permits it.

**Notebook uses the wrong Python environment** - Select the .venv
interpreter and choose the same environment as the Jupyter kernel. -
Restart the kernel after changing the interpreter or editing an imported
Python module, then rerun the notebook cells needed to rebuild
variables.

**Missing data or database tables** - Run the data-generation cell and
then the audit, cleaning and validation cells before running downstream
SQL, modelling, triage or agent sections.
### **Learning outcomes**
After working through the notebook, a learner should be able to explain
the data lifecycle from synthetic source generation through validation
and storage; basic RFM analysis and clustering; time-aware model
evaluation and leakage; privacy-aware ticket triage;
TF-IDF/cosine-similarity retrieval; tool-calling workflows; and safe
stopping and checkpoint/resume concepts.
