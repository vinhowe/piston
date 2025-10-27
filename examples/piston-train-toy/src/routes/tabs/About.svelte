<script lang="ts">
	import CuteLogo from '$lib/components/CuteLogo.svelte';
	import FN from '$lib/components/footnotes/Footnote.svelte';
	import FootnotesProvider from '$lib/components/footnotes/FootnotesProvider.svelte';
	import UserGuideTooltip from '$lib/components/UserGuideTooltip.svelte';
	import { browserInfo, hasWebGPU } from '$lib/workspace/ui.svelte';
	import { getIconStrokeWidth } from '$lib/workspace/ui.svelte';
	import { ExternalLink, Gpu } from 'lucide-svelte';

	const iconStrokeWidth = $derived(getIconStrokeWidth());
</script>

<div class="bg-neutral-100 h-full overflow-auto overscroll-contain flex flex-col flex-1 min-h-0">
	<div
		class="max-w-2xl mx-auto px-3 pb-3 @md:px-4 @md:pb-4 bg-white flex-1 flex flex-col justify-between"
	>
		<article class="prose text-base pb-13 [&_a]:underline">
			<FootnotesProvider start={1}>
				{#if !hasWebGPU.current}
					{@const isBrowserUnknown = browserInfo.current.type === 'unknown'}
					<UserGuideTooltip
						icon={Gpu}
						colorClass="border-yellow-400 bg-yellow-100 text-yellow-700 w-full"
						class="my-2 -mx-1 @md:-mx-2"
						role="alert"
					>
						<div class="mb-2">
							<p>
								<b>Sequence Toy requires WebGPU support, and your browser doesn't support it yet.</b
								>
								{#if isBrowserUnknown}
									You have a few options:
								{/if}
							</p>
						</div>
						<ul class="list-disc list-inside">
							{#if isBrowserUnknown || (browserInfo.current.type !== 'firefox' && browserInfo.current.type !== 'safari')}
								<li>
									Use <b>the latest version of Chrome</b>, or a
									<b>Chromium-based browser (Edge, Arc, Brave, etc.)</b>
									<ul class="list-disc list-inside ml-4">
										<li>
											If that doesn't work, try <b>Chrome Canary</b> and ensure WebGPU is enabled.
										</li>
									</ul>
								</li>
							{/if}
							{#if isBrowserUnknown || browserInfo.current.type === 'firefox'}
								<li>
									Use <a
										href="https://www.mozilla.org/en-US/firefox/new/"
										target="_blank"
										rel="noopener noreferrer"
										><b>Firefox Nightly</b>
										<ExternalLink
											class="inline-block h-3.5 w-3.5 -translate-y-0.5"
											strokeWidth={iconStrokeWidth}
										/></a
									>
								</li>
							{/if}
							{#if isBrowserUnknown || browserInfo.current.type === 'safari'}
								<li>
									Use the latest <b>Safari Technology Preview</b> or enable WebGPU via Feature
									Flags.
									<ul class="list-disc list-inside ml-4">
										{#if isBrowserUnknown || browserInfo.current.platform === 'ios'}
											<li>
												iOS: System Settings > Apps > Safari > Advanced > Feature Flags > Enable
												"WebGPU"
											</li>
										{/if}
										{#if isBrowserUnknown || browserInfo.current.platform === 'macos'}
											<li>
												MacOS: <a
													href="https://developer.apple.com/documentation/safari-developer-tools/feature-flag-settings"
													target="_blank"
													rel="noopener noreferrer"
													>Develop menu <ExternalLink
														class="inline-block h-3.5 w-3.5 -translate-y-0.5"
														strokeWidth={iconStrokeWidth}
													/></a
												> Feature Flags > Enable "WebGPU"
											</li>
										{/if}
									</ul>
								</li>
							{/if}
						</ul>
					</UserGuideTooltip>
				{/if}
				<div
					class="bg-purple-200 -mx-3 p-3 @md:-mx-4 @md:p-4 @md:mt-0 mb-3 @md:mb-4 space-y-3 @sm:space-y-5"
				>
					<div class="flex items-start w-full gap-2">
						{#each Array.from({ length: 8 }) as _, i (i)}
							<div style="flex: {(i + 2) ** 1.5};" class="flex justify-start">
								<CuteLogo class="w-[22px] h-[22px] text-purple-900" strokeWidth={1.7} />
							</div>
						{/each}
					</div>

					<h1
						class="text-[0.891rem] text-purple-900 uppercase leading-none font-mono font-[450] tracking-wide"
					>
						Sequence Toy
					</h1>
				</div>

				<p class="font-medium mb-4 mt-4 text-lg">
					Train a language model in your browser with WebGPU
				</p>

				<p class="mb-4">
					<a href="https://vin.how" target="_blank" rel="noopener noreferrer">Vin Howe</a>
					built
					<a href="https://github.com/vinhowe/piston" target="_blank" rel="noopener noreferrer"
						>Piston</a
					>, a WebGPU automatic differentiation engine with a JavaScript API modeled after PyTorch,
					for this project. It started life as a fork of
					<a href="https://github.com/huggingface/ratchet" target="_blank" rel="noopener noreferrer"
						>Ratchet</a
					><FN id="ratchet"
						>A ratchet is a device that allows motion in only one direction&mdash;Ratchet is
						intentionally forward-only. A piston reciprocates back and forth quickly&mdash;Piston
						adds support for reverse-mode automatic differentiation.</FN
					>
					by
					<a href="https://x.com/fleetwood___" target="_blank" rel="noopener noreferrer"
						>Christopher Fleetwood</a
					>.
				</p>

				<div class="space-y-4">
					<div>
						<h3 class="font-medium mb-2">
							This project was inspired by many other open-source projects:
						</h3>
						<div class="mb-4">
							<ul class="list-disc list-inside space-y-1">
								<li>
									<a href="https://github.com/0hq/WebGPT" target="_blank" rel="noopener noreferrer"
										>WebGPT</a
									>
									by
									<a href="https://x.com/willdepue" target="_blank" rel="noopener noreferrer"
										>Will Depue</a
									>.
								</li>
								<li>
									<a
										href="https://playground.tensorflow.org"
										target="_blank"
										rel="noopener noreferrer">TensorFlow Neural Network Playground</a
									>
									by
									<a href="https://x.com/dsmilkov" target="_blank" rel="noopener noreferrer"
										>Daniel Smilkov</a
									>
									and
									<a href="https://x.com/shancarter" target="_blank" rel="noopener noreferrer"
										>Shan Carter</a
									>.
								</li>
								<li>
									<a
										href="https://github.com/KellerJordan/modded-nanogpt"
										target="_blank"
										rel="noopener noreferrer">Modded-NanoGPT</a
									>
									by
									<a href="https://x.com/kellerjordan0" target="_blank" rel="noopener noreferrer"
										>Keller Jordan</a
									>.
								</li>
								<li>
									<a
										href="https://github.com/huggingface/transformers.js"
										target="_blank"
										rel="noopener noreferrer">Transformers.js</a
									>
									by
									<a href="https://x.com/xenovacom" target="_blank" rel="noopener noreferrer"
										>Joshua Lochner</a
									>.
								</li>
								<li>
									<a
										href="https://poloclub.github.io/transformer-explainer/"
										target="_blank"
										rel="noopener noreferrer">Transformer Explainer</a
									>
									by the
									<a href="https://x.com/polodataclub" target="_blank" rel="noopener noreferrer"
										>Polo Data Club at Georgia Tech</a
									>.
								</li>
								<li>
									<a href="https://bbycroft.net/llm" target="_blank" rel="noopener noreferrer"
										>LLM Visualization</a
									>
									by
									<a href="https://x.com/brendanbycroft" target="_blank" rel="noopener noreferrer"
										>Brendan Bycroft</a
									>.
								</li>
								<li>
									<a
										href="https://github.com/karpathy/convnetjs"
										target="_blank"
										rel="noopener noreferrer">ConvNetJS</a
									>,
									<a
										href="https://github.com/karpathy/micrograd"
										target="_blank"
										rel="noopener noreferrer">micrograd</a
									>,
									<a
										href="https://github.com/karpathy/minGPT"
										target="_blank"
										rel="noopener noreferrer">minGPT</a
									>, and
									<a
										href="https://github.com/karpathy/llm.c"
										target="_blank"
										rel="noopener noreferrer">llm.c</a
									>, by
									<a href="https://x.com/karpathy" target="_blank" rel="noopener noreferrer"
										>Andrej Karpathy</a
									>.
								</li>
							</ul>
						</div>
					</div>

					<div>
						<h3 class="font-medium mb-2">Helpful resources when building this project:</h3>
						<div class="mb-4">
							<ul class="list-disc list-inside space-y-1">
								<li>
									<a href="https://x.com/ezyang" target="_blank" rel="noopener noreferrer"
										>Edward Z. Yang</a
									>'s blog posts on
									<a
										href="https://blog.ezyang.com/2019/05/pytorch-internals/"
										target="_blank"
										rel="noopener noreferrer">PyTorch internals</a
									>
									and its
									<a
										href="https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/"
										target="_blank"
										rel="noopener noreferrer">dispatcher</a
									>.
								</li>
								<li>
									<a
										href="https://arxiv.org/abs/2102.13267"
										target="_blank"
										rel="noopener noreferrer"
									>
										The LazyTensor paper</a
									>
									and
									<a
										href="https://github.com/pytorch/pytorch/tree/main/torch/csrc/lazy"
										target="_blank"
										rel="noopener noreferrer">torch/csrc/lazy/</a
									>.
								</li>
								<li>
									Building PyTorch from source and spelunking in its source code&mdash;especially
									the generated parts.
								</li>
							</ul>
						</div>
					</div>

					<div>
						<div class="space-y-4">
							<div>
								<h4 class="font-medium mb-2">
									Thanks to <a
										href="https://x.com/grantpitt0"
										target="_blank"
										rel="noopener noreferrer">Grant Pitt</a
									>,
									<a href="https://x.com/fleetwood___" target="_blank" rel="noopener noreferrer"
										>Christopher Fleetwood</a
									>, and
									<a href="https://x.com/bgub_" target="_blank" rel="noopener noreferrer"
										>Ben Gubler</a
									>
									for support, discussion, and feedback. 💜
								</h4>
							</div>
						</div>
					</div>
				</div>
			</FootnotesProvider>
		</article>
		<footer class="relative flex justify-between items-center text-neutral-600 py-1">
			<span>
				By <a href="https://vin.how" class="underline" target="_blank" rel="noopener noreferrer"
					>Vin Howe</a
				>
			</span>
		</footer>
	</div>
</div>
