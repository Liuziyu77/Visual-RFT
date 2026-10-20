# Evaluation on MAT-Search
In this folder, we provide scripts for evaluation on **MAT-Search** and other four multihopQA Benchmark (2wikimultihopQA, HotpotQA, MuSiQue, Bamboogle).
``MAT-Search-Benchmark`` folder is used to evaluate the results on **MAT-Search**, while `traditional_benchmark` folder is used to evaluate the inference results of the other four multihopQA Benchmark.

## Download Dataset and Models
You can download MAT Benchmakr on 🤗<a href="https://huggingface.co/datasets/laolao77/MAT">Datasets</a>. Use `/MAT/MAT-Benchmark/MAT-Search.json`.

You can download our model: 🤗<a href="https://huggingface.co/laolao77/Visual-ARFT-Search">Visual-ARFT-Search</a></h3>

You can download other four multihopQA benchmark from 🤗<a href="https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets">Dataset</a></h3>.

## Web Search API
The Visual-ARFT evaluation agents use [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
for free web search in Fast mode when no search provider or Serper key is configured.
No Parallel account or API key is needed. Install the search dependencies in your
Qwen2.5-VL inference environment, from this directory:

```bash
python -m pip install -r requirements-search.txt
```

A real `SERPER_API` environment variable or `.env` key keeps Serper selected.
The dummy `xxxxxxxxxxxxxxxxxxxxxxxxx` value shipped in `.env` is treated as unconfigured.
To reproduce the paper's Serper-based evaluation, set that key and use Serper.
Search providers can return different evidence, so changing providers can change
benchmark results. The original Serper and DuckDuckGo helper functions remain
available, including their Firecrawl `pro` mode.

Set `VISUAL_ARFT_SEARCH_PROVIDER` in the environment or `.env` to explicitly choose
`parallel`, `serper`, `ddg`, or `disabled`. Explicit choices take precedence over
credential detection; provider errors never fall back to another service.
`serper` requires `SERPER_API`. For `ddg`, additionally install `duckduckgo-search`;
the optional `pro` helpers additionally require `firecrawl-py` and `FIRECRAWL_API`.
`disabled` prevents search requests and marks questions that request a search as
unanswered. A question that answers directly can still complete.

Each evaluation question has its own opaque search session ID, reused across
follow-up searches. Parallel results are adapted to the existing `title`/`href`/`body`
format, with up to four results fed into the agent's next `<information>` block.
Requests have a 30-second timeout; an error or free-service quota response leaves
the affected question unanswered. Interrupting the evaluation cancels the active
Parallel request. These settings apply to the five evaluation agent scripts here;
the notebook demos and offline training keep their existing setup.

The search integration can be checked without downloading model weights:

```bash
python -m unittest discover -s tests -v
```

These tests run the production evaluation loops with scripted model responses and
an in-memory MCP HTTP transport. They check search dispatch and prompt consumption,
not model quality or full GPU benchmark scores.

## Inference on MAT-Search

Firstly, `cd MAT-Search-Benchmark`. In this folder, there are `evaluation_mat_search_ngpu_7b_df.py` and `evaluation_mat_search_ngpu_7b_visual_arft.py`. The first Python file is used to evaluate the results of direct inference, while the second Python file is used to evaluate the results of the **Visual-ARFT-Search** model.

To run `evaluation_mat_coding_visual_arft.py`, you need to replace the paths to the model and dataset:

-  Line 59: Replace **model_name** with the actual model path (**Visual-ARFT-Search**).
-  Line 76: Replace **json_path** with the actual dataset path (MAT-Search.json).
-  Line 97: Replace **input_image_path** with the actual image path.
-  Line 192: Set the results save path.

To run `evaluation_mat_search_ngpu_7b_df.py`, you need to replace the paths to the model and dataset:

-  Line 17: Replace **model_name** with the actual model path (**Original Qwen2.5-VL without Visual-ARFT**).
-  Line 27: Replace **json_path** with the actual dataset path (MAT-Search.json).
-  Line 37: Replace **input_image_path** with the actual image path.
-  Line 95: Set the results save path.

> ⏳ The code `evaluation_mat_coding_visual_arft.py` supports multi-GPU execution. The inference time for **Visual-ARFT-Coding-3/7B** is around ten minutes with 8GPUs.

## Inference on other MultihopQA Benchmark

Firstly, `cd traditional_benchmark`. Then, replace the paths as instructed above.

**2Wiki** contains over 12k questions and **HotpotQA** has over 7k, so the inference process may take a relatively long time. However, all scripts in the `traditional_benchmark` folder support multi-GPU inference.

## Evaluation
After obtaining the inference results, run the `evaluation.ipynb` in each folder step by step. The `.ipynb` file will provide the final evaluation scores.

We have also saved the inference results of the `Visual-ARFT-Search-7B` model in the `evaluation_results` folder. You can directly use the results inside to test the evaluation scores.







