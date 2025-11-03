<script lang="ts">
	import type { Snippet } from 'svelte';

	import KatexBlock from '../KatexBlock.svelte';

	type $$Props = {
		label: string;
		children: Snippet;
		funFact?: Snippet;
	};

	let { label, children, funFact }: $$Props = $props();
</script>

<!-- class="px-1 py-0.5 bg-gradient-to-b from-neutral-200 via-neutral-300/5 to-transparent border border-neutral-400/80 flex flex-col justify-between text-neutral-800" -->
<div
	class="px-1 py-0.5 bg-neutral-100/70 border border-neutral-400/60 flex flex-col justify-between text-neutral-800"
>
	<span class="font-bold text-neutral-500 uppercase text-2xs font-mono tracking-wider"
		><KatexBlock text={label} /></span
	>
	<span class="flex items-baseline gap-x-1">
		<span class="font-mono font-semibold text-controls-numeric">
			{@render children?.()}
		</span>
		{#if funFact}
			<span class="font-sans text-right w-full text-xs font-medium text-neutral-600">
				{@render funFact?.()}
			</span>
		{/if}
	</span>
</div>
