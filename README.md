<div align="center">
<h1>Sequence Toy + Piston</h1>
<a href="https://sequence.toys">
<img width="700px" src=".github/media/sequence-toy-hero.webp">
</a>
<p><a href="https://sequence.toys">Web Playground</a> &bull; <a href="https://vin.how/blog/train-a-language-model-in-browser">Blog post</a></p>
<p align="center">
Train language models in your browser with WebGPU-powered autodiff.
</p>
<br>
</div>

Sequence Toy is a web playground for training sequence models (Transformers, LSTMs, GRUs, and vanilla RNNs) using Piston, a proof-of-concept WebGPU automatic differentiation library. This repository houses these related projects.

## Attribution

- Piston is a fork of [Ratchet](https://github.com/huggingface/ratchet), hacked and butchered to add automatic differentiation. I picked Ratchet because it is simple enough to reason about, but it thoughtfully supports WebGPU via [wgpu](https://github.com/gfx-rs/wgpu).
- My implementation of backprop borrows heavily from [Candle](https://github.com/huggingface/candle).
- The lazy execution model is an implementation of [LazyTensor](https://arxiv.org/abs/2102.13267), and borrows from [torch/csrc/lazy](https://github.com/pytorch/pytorch/tree/main/torch/csrc/lazy).
- I used [Keller Jordan](https://x.com/kellerjordan0)'s [Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt) as a reference for his Muon optimizer.
- My BPE tokenizer implementation is simplified from [transformers.js](https://github.com/huggingface/transformers.js), which in turn mirrors [transformers](https://github.com/huggingface/transformers).
- My GPT-2 model implementation was originally based on [minGPT](https://github.com/karpathy/minGPT).
- I adapted dataset preprocessing code from [llm.c](https://github.com/karpathy/llm.c).
