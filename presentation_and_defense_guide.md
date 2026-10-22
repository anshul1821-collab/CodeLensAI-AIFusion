# CodeLens AI — Presentation Script & Judge Defense Guide

---

## 🎙️ 4-to-5 Minute Presentation Script

### [0:00 - 0:45] Introduction & The Core Problem
"Good morning, judges. Every day, thousands of computer science students and developers struggle with unfamiliar code and mysterious runtime bugs. Today, students paste their code into generic AI chatbots. But generic chatbots suffer from two fatal flaws: **hallucination** and **lack of documentation grounding**. They invent explanations, cite non-existent documentation, or guess without evidence.

Our mission with **CodeLens AI** is captured in our tagline:
> *'Understand your code. Find the bug. Trust the evidence.'*

CodeLens AI is a context-grounded programming code explainer and debugging assistant built on a real Retrieval-Augmented Generation (RAG) architecture with Django."

---

### [0:45 - 1:45] Architectural Innovation: Real RAG Pipeline
"Unlike generic chat wrappers, CodeLens AI enforces **grounding before generation**:
1. **Document Ingestion**: Users can upload official documentation and course notes in PDF and TXT formats. Our system extracts text page-by-page while preserving exact page and line metadata.
2. **Boundary-Aware Chunking**: We segment text at semantic paragraph and sentence boundaries with configurable chunk size and overlap, ensuring context is never truncated awkwardly mid-sentence.
3. **Mathematical Vector Embeddings**: We calculate real vector embeddings for every document chunk and query. Our system supports Google Gemini's `text-embedding-004` alongside our zero-dependency 512-dimensional deterministic semantic vector engine.
4. **Persistent Vector Store & Cosine Similarity**: Embeddings are stored persistently in our SQLite vector store. When a query arrives, we compute exact cosine similarity scores:
   $$\text{sim}(q, d) = \frac{\sum q_i d_i}{\|q\| \|d\|}$$
   and rank matching passages against a strict relevance threshold.
5. **Honest Grounding Limitation**: If the knowledge base does not contain relevant documentation, CodeLens refuses to hallucinate and explicitly acknowledges the limitation."

---

### [1:45 - 3:00] Live Demonstration Walkthrough
*(Presenter transitions to browser running `http://127.0.0.1:8000/`)*

"Let's look at a live demonstration.
Here on the screen is CodeLens AI's developer-focused dark interface.
In our Knowledge Base, you can see **4 authentic reference documents** and **30 chunks** already indexed, including a 2-page PDF and 3 official Python documentation guides.

Let's select the classic student bug: **Mutable Default Arguments**.
Look at this code:
```python
def add_student_grade(student_name, grade_list=[]):
    grade_list.append(student_name)
    return grade_list
```
The student asks: *'Why does class_b contain Alice when called with Bob?'*

Let's click **'Analyze Code & Retrieve Docs'**.

Notice what happens:
1. **Retrieval Occurred First**: Look at the right panel. Before writing the explanation, CodeLens queried our vector store and retrieved **Source #1** from Python Tutorial Section 4.8.1 with a **64% relevance score**.
2. **Verifiable Citations**: Under Potential Bugs, CodeLens highlights:
   *'Mutable Default Argument in add_student_grade()'* with severity **Critical**, explaining that default argument expressions are evaluated once at function definition time, not on each invocation.
3. **Suggested Correction**: It provides the standard Python idiomatic fix using `grade_list=None`, and outputs complete, verified corrected code ready to copy with one click.
4. **Evidence Inspection**: If the student wants proof, they click **'View Full Passage'** on Source #1 to read the verbatim documentation explaining the underlying memory model."

---

### [3:00 - 3:45] Security & Production Readiness
"Security was a first-class citizen in our design:
- **Zero Untrusted Code Execution**: We analyze code through static AST inspection and lexical pattern matching. CodeLens never uses `eval()`, `exec()`, or subprocess execution.
- **CSRF & Upload Protection**: Full CSRF protection, file type whitelisting, and a 10MB memory upload cap.
- **Fail-Safe Reliability**: If an external API is unconfigured or experiences network downtime, CodeLens seamlessly runs on its built-in local vector engine without crashing."

---

### [3:45 - 4:30] Conclusion & Impact
"In summary, CodeLens AI answers the four questions every student asks:
1. What does this code do?
2. Why is it failing?
3. How do I fix it?
4. Which official documentation proves it?

Our test suite confirms 16 out of 16 unit and integration tests passing. CodeLens AI empowers developers and students to write better code with confidence and evidence. Thank you, and we welcome your questions!"

---

## 🎯 5 Likely Judge Questions & Winning Technical Answers

### Question 1: "How do you ensure that your RAG pipeline is performing real retrieval rather than mocking or hardcoding source cards?"
**Answer:**
> "Our retrieval pipeline is 100% dynamic and mathematically grounded. When a document is uploaded, `assistant.rag.extractor` extracts raw text page-by-page. `assistant.rag.chunker` splits it into overlapping segments with page metadata. `assistant.rag.embedder` computes real 512-dimensional vector representations that are stored in the `DocumentChunk` table. During analysis, we synthesize a semantic query, embed it using the exact same vector model, compute the dot product and cosine similarity against all chunks in the database, and filter using a minimum threshold. If you delete a document from the Knowledge Base tab, its chunks are deleted immediately and will no longer appear in retrieval. You can also test this live in our Vector Search Inspector tab by typing any arbitrary query."

---

### Question 2: "How does CodeLens handle out-of-domain queries or when the knowledge base lacks relevant information?"
**Answer:**
> "Unlike general-purpose chatbots that hallucinate plausible-sounding explanations, CodeLens implements a strict **Grounding Threshold Validator** (`min_similarity = 0.18 - 0.22`). If the top similarity score is below the threshold or the knowledge base is empty, `is_grounded` is flagged as `False`. The UI immediately displays an amber warning card explicitly acknowledging:
> *'I couldn't find sufficiently relevant information in your current knowledge base. Upload appropriate programming documentation or add a relevant reference before requesting a documentation-grounded explanation.'*
> While static structural notes may be provided, CodeLens never claims that an answer is documentation-supported unless real matching evidence was retrieved."

---

### Question 3: "Why did you choose Django for the backend instead of a lightweight framework like FastAPI or Flask?"
**Answer:**
> "Django was chosen for three concrete engineering reasons:
> 1. **Robust ORM & Persistent Storage**: Django’s ORM and migrations allow us to persist documents, chunk embeddings, and analysis audit records in SQLite cleanly with full schema integrity and cascade deletions.
> 2. **Built-in Security Defaults**: Django provides battle-tested CSRF protection, session management, secure file upload handlers, and strict request validation out of the box.
> 3. **Monolithic Reliability**: By utilizing Django templates, Bootstrap 5, and vanilla JavaScript, we eliminate complex Node build steps and fragile frontend-backend synchronization, keeping the system maintainable for a solo developer within a tight hackathon window while meeting the official problem statement specifications."

---

### Question 4: "How does your local vector embedder work without native libraries like FAISS or heavy PyTorch models?"
**Answer:**
> "In environments where C++ build tools or GPU acceleration are unavailable (such as modern Windows Python 3.14 environments where prebuilt wheels for FAISS/PyMuPDF are not yet published), our `LocalSemanticEmbedder` provides a high-dimensional, deterministic semantic vector model:
> - It tokenizes input into unigrams, bigrams, and subword 3-gram and 4-gram character features to capture morphology and syntax.
> - It utilizes **Semantic Concept Synset clustering**, mapping synonymous programming terms (such as `parameter` $\leftrightarrow$ `argument`, `mutable` $\leftrightarrow$ `list`) into shared semantic concept dimensions.
> - It applies $L_2$ normalization so that cosine similarity is computed directly via dot products.
> This yields high, stable similarity scores for conceptually related code and documentation while maintaining near-zero similarity for unrelated queries."

---

### Question 5: "What security measures prevent malicious code execution if an attacker submits dangerous scripts like `os.system('rm -rf /')`?"
**Answer:**
> "CodeLens AI enforces a strict **Zero-Execution Sandbox Policy**:
> 1. Submitted code is treated purely as untrusted text data. It is never passed to Python's `eval()`, `exec()`, or any subprocess.
> 2. Code parsing is done via Python's built-in `ast.parse()`, which produces an Abstract Syntax Tree for lexical and structural inspection without executing any bytecode.
> 3. Our test suite includes a dedicated canary security test (`test_untrusted_code_is_never_executed`) that submits code attempting to write an unauthorized file to the filesystem; the test verifies that the file is never created.
> 4. All rendered code in the interface is strictly HTML-escaped before display to prevent Cross-Site Scripting (XSS)."
