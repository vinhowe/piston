For the time being, I don't have instructions on how to run this. You're on your own for a tiny bit :)

<div align="center">
<img width="200px" src="https://github.com/vinhowe/ratchet-backward/raw/master/.github/ratchet.png">
<p align="center">
<code>(backward)</code>
</p>
<br>
</div>

This is a fork of [Ratchet](https://github.com/huggingface/ratchet), hacked and butchered to add backpropogation, to show that it is technically possible to train language models in a (WebGPU-enabled) browser.

Currently this project is a proof-of-concept—it is possible to slowly train a small GPT-2-like language model, in Chrome, from scratch, on an M1 Pro MacBook.

I picked Ratchet because it is simple enough to reason about, but it thoughtfully supports WebGPU's lazy execution model. Much of my implementation beyond Ratchet is adapted from the [Candle](https://github.com/huggingface/candle) project.
