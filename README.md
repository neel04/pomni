# Pomni

Current Frontier LLMs are really bad at assisting with writing JAX code due to the sparse resources spread across multiple different platforms.

Pomni is a finetune of a gemma model on JAX data, meant to run locally. It provides a TUI interface making it easy to integrate into existing terminal workflows.


## Setup

To use the finetuned pomni model, you can simply do:

```bash
pip install pomni
pomni
```

to automatically download and run the model.


## Screenshot

![Pomni TUI](https://raw.githubusercontent.com/nee04/pomni/main/tui/pomni_tui_ss.jpeg)
