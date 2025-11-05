# Sequence Toy + Piston library

Train small sequence models in your browser with WebGPU.

## Attribution

Piston is a fork of [Ratchet](https://github.com/huggingface/ratchet), hacked and butchered to add backpropogation, to show that it is technically possible to train language models in a WebGPU-enabled browser.

I picked Ratchet because it is simple enough to reason about, but it thoughtfully supports WebGPU's lazy execution model. Much of my implementation beyond Ratchet is adapted from the [Candle](https://github.com/huggingface/candle) project.
