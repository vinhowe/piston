<script lang="ts">
	import { decodeSingle } from '$lib/train/tokenizer';
	import { getCurrentDataset } from '$lib/workspace/config.svelte';

	const { sampleData, tokenizer, dataset } = $derived(
		getCurrentDataset() ?? { sampleData: [], tokenizer: null, dataset: null }
	);

	//
	// BEGIN involved thing we do to avoid height jittering when switching between datasets or tokenizers
	//

	let tokenizerPending = $state(false);
	let containerHeight = $state(0);
	let lastStableHeight = $state(0);
	let sampleDataPending = $state(false);
	const anyPending = $derived(tokenizerPending || sampleDataPending);

	let currentTokenizerPromise: Promise<unknown> | null = $state(null);
	let currentSampleDataPromise: Promise<unknown> | null = $state(null);

	$effect(() => {
		if (tokenizer instanceof Promise && tokenizer !== currentTokenizerPromise) {
			if (containerHeight > 0) {
				lastStableHeight = containerHeight;
			}
			currentTokenizerPromise = tokenizer;
			tokenizerPending = true;
			tokenizer.finally(() => {
				if (currentTokenizerPromise === tokenizer) {
					tokenizerPending = false;
				}
			});
		} else if (!(tokenizer instanceof Promise)) {
			currentTokenizerPromise = null;
			tokenizerPending = false;
		}
	});

	$effect(() => {
		if (sampleData instanceof Promise && sampleData !== currentSampleDataPromise) {
			if (containerHeight > 0) {
				lastStableHeight = containerHeight;
			}
			currentSampleDataPromise = sampleData;
			sampleDataPending = true;
			sampleData.finally(() => {
				if (currentSampleDataPromise === sampleData) {
					sampleDataPending = false;
				}
			});
		} else if (!(sampleData instanceof Promise)) {
			currentSampleDataPromise = null;
			sampleDataPending = false;
		}
	});

	$effect(() => {
		if (!anyPending && containerHeight > 0) {
			lastStableHeight = containerHeight;
		}
	});

	//
	// END involved thing
	//

	function maskedFlagsForRange(
		fullSequence: number[] | undefined,
		startIndex: number,
		length: number
	): boolean[] {
		const maskId = dataset?.maskId;
		if (typeof maskId !== 'number' || !Array.isArray(fullSequence)) {
			return new Array(length).fill(false);
		}
		return fullSequence.slice(startIndex, startIndex + length).map((t) => t === maskId);
	}
</script>

{#snippet token(value: number, dashed: boolean = false)}
	<span
		class={`text-black px-0.5 border shrink-0 ${dashed ? 'border-neutral-700 border-dashed z-10' : 'border-neutral-300'}`}
	>
		{#if tokenizer instanceof Promise}
			{#await tokenizer then tokenizer}
				{decodeSingle(value, tokenizer)}
			{/await}
		{:else}
			{decodeSingle(value, tokenizer)}
		{/if}
	</span>
{/snippet}

{#snippet tokenSequence(values: number[], ignored?: boolean[])}
	<div class="w-full flex -space-x-px">
		{#each values as value, i (i)}
			{@render token(value, ignored ? ignored[i] : false)}
		{/each}
	</div>
{/snippet}

<div
	class="border-x border-panel-border-base overflow-x-auto"
	bind:clientHeight={containerHeight}
	style:height={anyPending && lastStableHeight > 0 ? `${lastStableHeight}px` : 'auto'}
	style:overflow-y={anyPending ? 'hidden' : 'visible'}
>
	{#await sampleData then { collated, hasPrompt }}
		{#if collated.length > 0}
			<div class="overflow-x-auto -mx-px">
				<table
					class="w-full text-sm border-collapse [&_th]:border-panel-border-base [&_th]:border [&_td]:border-panel-border-base [&_td]:border"
				>
					<thead>
						<tr
							class="border-b bg-neutral-100 [&_th]:font-medium [&_th]:border-r [&_th]:border-b text-neutral-700 [&_th]:px-1 [&_th]:py-0.5 [&_th]:text-left"
						>
							{#if hasPrompt}
								<th>Prompt</th>
							{/if}
							<th>Target</th>
						</tr>
					</thead>
					<tbody>
						{#each collated as { prompt, target, fullSequence } (Array.prototype.concat.call(fullSequence, prompt ?? [], target ?? []))}
							{@const pLen = prompt?.length || 0}
							{@const targetFlags = maskedFlagsForRange(fullSequence, pLen, target?.length ?? 0)}
							<tr
								class="border-b border-neutral-100 last:border-b-0 [&_td]:p-1 font-mono text-controls-numeric"
							>
								{#if hasPrompt}
									{@const promptFlags = maskedFlagsForRange(fullSequence, 0, pLen)}
									<td>
										{@render tokenSequence(prompt!, promptFlags)}
									</td>
								{/if}
								<td>
									{@render tokenSequence(target ?? fullSequence, targetFlags)}
								</td>
							</tr>
						{/each}
						<tr
							class="border-b border-neutral-100 last:border-b-0 [&_td]:p-1 font-mono text-controls-numeric"
						>
							{#if hasPrompt}
								<td>...</td>
							{/if}
							<td>...</td>
						</tr>
					</tbody>
				</table>
			</div>
		{/if}
	{/await}
</div>
