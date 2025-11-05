<script lang="ts">
	import CuteLogo from '$lib/components/CuteLogo.svelte';
	import FN from '$lib/components/footnotes/Footnote.svelte';
	import FootnotesProvider from '$lib/components/footnotes/FootnotesProvider.svelte';
	import UserGuideTooltip from '$lib/components/UserGuideTooltip.svelte';
	import { config, setPreset, validateConfig } from '$lib/workspace/config.svelte';
	import {
		browserInfo,
		hasWebGPU,
		isVisualizerEditorMinimized,
		startTraining
	} from '$lib/workspace/ui.svelte';
	import { getIconStrokeWidth } from '$lib/workspace/ui.svelte';
	import { openConfigAndScrollToControl, switchToMetrics } from '$lib/workspace/ui.svelte';
	import { getVisualizationExampleById } from '$lib/workspace/visualizationExamples';
	import { trainingState, updateVisualizerScript } from '$lib/workspace/workers.svelte';
	import { ExternalLink, Gpu } from 'lucide-svelte';
	import { onMount } from 'svelte';

	const iconStrokeWidth = $derived(getIconStrokeWidth());

	let expanded = $state(false);
	let isOverflowing = $state(false);
	let contentRef: HTMLDivElement;

	const COLLAPSED_MAX_HEIGHT = 100;

	function measureOverflow() {
		if (contentRef) {
			isOverflowing = contentRef.scrollHeight > COLLAPSED_MAX_HEIGHT + 1;
		}
	}

	function wrapClickHandlerForKeyboard(onClick: () => void) {
		return (e: KeyboardEvent) => {
			if (e.key === 'Enter' || e.key === ' ') {
				e.preventDefault();
				onClick();
			}
		};
	}

	onMount(() => {
		measureOverflow();
		const ro = new ResizeObserver(() => measureOverflow());
		if (contentRef) ro.observe(contentRef);
		const onResize = () => measureOverflow();
		window.addEventListener('resize', onResize);
		return () => {
			ro.disconnect();
			window.removeEventListener('resize', onResize);
		};
	});
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

				<div class="mb-4 -mx-2">
					<div
						class=" bg-purple-50 p-2 leading-relaxed border border-purple-500 border-dashed [&_b]:font-medium"
					>
						<h2 class="font-medium">You could try&hellip;</h2>
						<div class="relative">
							<div
								bind:this={contentRef}
								class="overflow-hidden transition-[max-height] duration-300"
								style={`max-height: ${expanded || !isOverflowing ? 'none' : COLLAPSED_MAX_HEIGHT + 'px'}`}
							>
								{#snippet sClick(text: string, onClick: () => void)}
									{@const _onClick = onClick}
									<b
										class="cursor-pointer underline-offset-2 decoration-dotted underline font-medium text-purple-700"
										role="button"
										tabindex="0"
										onkeydown={wrapClickHandlerForKeyboard(_onClick)}
										onclick={_onClick}>{text}</b
									>
								{/snippet}
								{#snippet sPreset(text: string, id: string, controlId?: string)}
									{@render sClick(text, () => {
										setPreset(id);
										if (controlId) openConfigAndScrollToControl(controlId, ['task']);
									})}
								{/snippet}
								{#snippet sToyPreset(text: string, id: string, controlId?: string)}
									{@render sClick(text, () => {
										setPreset(id);
										config.model.family = 'transformer';
										validateConfig();
										if (controlId) openConfigAndScrollToControl(controlId, ['task']);
									})}
								{/snippet}
								{#snippet sViz(text: string, exampleId: string, expandEditor: boolean = false)}
									{@render sClick(text, () => {
										config.visualization.example = exampleId;
										config.visualization.script = null;
										switchToMetrics();
										if (trainingState.current !== 'stopped') {
											const script = getVisualizationExampleById(exampleId).script;
											updateVisualizerScript(exampleId, script);
										}
										startTraining();
										if (expandEditor) {
											isVisualizerEditorMinimized.current = false;
										}
									})}
								{/snippet}
								{#snippet sModelTransform(text: string, controlId: string)}
									{@render sClick(text, () => {
										config.model.family = 'transformer';
										validateConfig();
										openConfigAndScrollToControl(controlId, ['model']);
									})}
								{/snippet}
								{#snippet sModelRnn(text: string, controlId: string)}
									{@render sClick(text, () => {
										config.model.family = 'rnn';
										validateConfig();
										openConfigAndScrollToControl(controlId, ['model']);
									})}
								{/snippet}
								{#snippet sModelRnnLstm(text: string, controlId: string)}
									{@render sClick(text, () => {
										config.model.family = 'rnn';
										config.model.rnn.cellType = 'lstm';
										validateConfig();
										openConfigAndScrollToControl(controlId, ['model']);
									})}
								{/snippet}
								{#snippet sScrollTraining(text: string, controlId: string)}
									{@render sClick(text, () => {
										openConfigAndScrollToControl(controlId, ['training']);
									})}
								{/snippet}
								{#snippet sScrollOptimizer(text: string, controlId: string)}
									{@render sClick(text, () => {
										openConfigAndScrollToControl(controlId, ['optimizer']);
									})}
								{/snippet}
								{#snippet sEncoderBidirectional(text: string)}
									{@render sClick(text, () => {
										if (config.model.topology === 'decoder') config.model.topology = 'encoder';
										validateConfig();
										openConfigAndScrollToControl(
											'model-rnn-bidirectional-encoder-checkbox',
											['model'],
											{
												useLabelFor: true
											}
										);
									})}
								{/snippet}
								{#snippet sEnsureEncDec(text: string)}
									{@render sClick(text, () => {
										config.model.family = 'rnn';
										config.model.topology = 'encoder-decoder';
										validateConfig();
										openConfigAndScrollToControl('model-rnn-encoder-decoder-attention-group', [
											'model'
										]);
									})}
								{/snippet}
								<p>
									&hellip;training a
									<!-- <span
										class="inline-flex align-baseline border border-purple-300/80 overflow-hidden select-none p-px text-sm"
									>
										{#snippet modelTypeButton(
											type: 'decoder' | 'encoder-decoder',
											displayName: string
										)}
											<button
												type="button"
												class={`px-0.75 py-0.25 cursor-pointer leading-none border ${aboutToyTopology === type ? 'bg-purple-200 border-purple-400' : 'bg-transparent text-purple-800 border-transparent'}`}
												onclick={() => {
													aboutToyTopology = type;
													config.model.topology = type;
													validateConfig();
												}}
												aria-pressed={aboutToyTopology === type}
												aria-label={`Use ${displayName}`}
											>
												{displayName}
											</button>
										{/snippet}
										{@render modelTypeButton('encoder-decoder', 'Transformer')}
										{@render modelTypeButton('decoder', 'GPT')}
									</span> -->Transformer<FN
										><a
											href="https://arxiv.org/abs/1706.03762"
											target="_blank"
											rel="noopener noreferrer">Attention is All You Need</a
										></FN
									>
									to&#32;{@render sToyPreset(
										'sort characters',
										'sort-characters',
										'dataset-control'
									)},
									{@render sToyPreset('reverse a sequence', 'reverse-sequence', 'dataset-control')},
									or
									{@render sToyPreset(
										'find the numbers that add up to a sum',
										'two-sum',
										'dataset-control'
									)}, then {@render sModelRnnLstm(
										'compare with an LSTM',
										'rnn-cell-type-control'
									)}<FN
										><a
											href="https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf"
											target="_blank"
											rel="noopener noreferrer">Long Short-Term Memory</a
										></FN
									>.
									<b>Visualize</b>
									{@render sViz('the gradients in the attention layer', 'attention-gradients')}, {@render sViz(
										'all parameters in the network',
										'all-parameters'
									)}, or try
									{@render sViz('writing your own visualization queries', 'tutorial', true)}.
									<b>Learn to</b>
									{@render sPreset(
										'match parentheses in a Dyck language',
										'dyck-encoder',
										'dataset-control'
									)}<FN
										><a
											href="https://en.wikipedia.org/wiki/Dyck_language"
											target="_blank"
											rel="noopener noreferrer">Wikipedia article on Dyck languages</a
										></FN
									>
									using an encoder-only masked language modeling (MLM) objective<FN
										><a
											href="https://arxiv.org/abs/1810.04805"
											target="_blank"
											rel="noopener noreferrer"
											>BERT: Pre-training of Deep Bidirectional Transformers for Language
											Understanding</a
										></FN
									>
									<!--(cite masked modeling)-->.
									<b>Want a taste of natural language?</b>
									Try {@render sPreset(
										'training a GPT on TinyStories',
										'tinystories',
										'dataset-control'
									)}<FN
										><a
											href="https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf"
											target="_blank"
											rel="noopener noreferrer"
											>GPT: Improving Language Understanding by Generative Pre-Training</a
										></FN
									><FN
										><a
											href="https://arxiv.org/abs/2305.07759"
											target="_blank"
											rel="noopener noreferrer"
											>TinyStories: How Small Can Language Models Be and Still Speak Coherent
											English?</a
										></FN
									>, a dataset of short stories generated by GPT-4—and try different tokenizer
									sizes.
									<b>Play with</b>
									{@render sModelTransform('attention gating', 'model-attention-gating-group')},
									{@render sModelTransform('MLP variants', 'model-mlp-group')}, {@render sScrollOptimizer(
										'learning rate schedulers',
										'optimizer-lr-scheduler-group'
									)},
									{@render sModelTransform('initialization', 'model-initialization-group')}, {@render sScrollTraining(
										'dropout',
										'training-dropout-group'
									)}, or {@render sModelTransform(
										'QK normalization',
										'model-transformer-normalization-qk-norm-group'
									)}.
									<b>Want to train an RNN?</b>
									Mess with
									{@render sModelRnn('layer normalization', 'model-rnn-layer-normalization-group')},
									{@render sModelRnn('initialization', 'model-initialization-group')},
									{@render sEncoderBidirectional('bidirectional encoder layers')}, {@render sEnsureEncDec(
										'encoder-decoder attention variants'
									)}, or
									{@render sScrollTraining(
										'gradient norm clipping',
										'training-clip-grad-norm-group'
									)}.
									<b
										>Have a lot of VRAM and want to try something untested<FN label="*"
											>Vin, who owns a M1 Pro Macbook with 16GB of unified memory, has never been
											able to do this without running out of memory. Good luck!</FN
										>?</b
									>
									Try
									{@render sPreset(
										'training a GPT-2-sized model on FineWeb',
										'fineweb',
										'dataset-control'
									)}<FN
										><a
											href="https://arxiv.org/abs/2406.17557"
											target="_blank"
											rel="noopener noreferrer"
											>The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale</a
										></FN
									> (and
									<a
										href="https://x.com/messages/compose?recipient_id=1286805122115280896"
										target="_blank"
										rel="noopener noreferrer">DM me</a
									> if you get it to work!).
								</p>
							</div>
							{#if isOverflowing && !expanded}
								<div
									class="pointer-events-none absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-b from-transparent to-purple-50"
								></div>
							{/if}
						</div>
						{#if isOverflowing}
							<div class:mt-1={expanded}>
								<button
									type="button"
									class="text-purple-700 underline decoration-dotted underline-offset-2 font-medium text-sm select-none cursor-pointer"
									onclick={() => (expanded = !expanded)}
									aria-expanded={expanded}
								>
									{#if expanded}Fewer ideas{/if}{#if !expanded}More ideas{/if}
								</button>
							</div>
						{/if}
					</div>
				</div>

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
			<CuteLogo class="w-[22px] h-[22px]" strokeWidth={1.5} />
			<span>
				Revision <a
					href={`https://github.com/vinhowe/piston/commit/${__COMMIT_HASH__}`}
					class="underline"
					target="_blank"
					rel="noopener noreferrer">{__COMMIT_HASH__}</a
				>
			</span>
		</footer>
	</div>
</div>
