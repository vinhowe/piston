<div align="center">
<h1>Sequence Toy + Piston library</h1>
<a href="https://sequence.toys">
<img src=".github/media/sequence-toy-hero.webp">
</a>
<p><a href="https://sequence.toys">Web Playground</a></p>
<p align="center">
Train language models in your browser with WebGPU-powered autodiff.
</p>
<br>
</div>

Sequence Toy is a web playground for training sequence models (Transformers, LSTMs, GRUs, and vanilla RNNs) using Piston, a proof-of-concept WebGPU automatic differentiation library. This repository houses these related projects.

## Attribution

Piston is a fork of [Ratchet](https://github.com/huggingface/ratchet), hacked and butchered to add automatic differentiation. I picked Ratchet because it is simple enough to reason about, but it thoughtfully supports WebGPU's lazy execution model. Much of my implementation beyond Ratchet is adapted from the [Candle](https://github.com/huggingface/candle) project.
