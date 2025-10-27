<script lang="ts">
	import { decodeSingle } from '$lib/train/tokenizer';
	import { getCurrentDataset } from '$lib/workspace/config.svelte';

	const {
		sampleData: { collated, hasPrompt },
		tokenizer,
		dataset
	} = $derived(getCurrentDataset() ?? { sampleData: [], tokenizer: null, dataset: null });

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
		{decodeSingle(value, tokenizer)}
	</span>
{/snippet}

{#snippet tokenSequence(values: number[], ignored?: boolean[])}
	<div class="w-full flex -space-x-px">
		{#each values as value, i (i)}
			{@render token(value, ignored ? ignored[i] : false)}
		{/each}
	</div>
{/snippet}

<div class="border-x border-panel-border-base overflow-x-auto">
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
</div>
