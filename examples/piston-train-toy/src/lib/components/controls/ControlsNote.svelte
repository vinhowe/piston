<script lang="ts">
	import type { Snippet } from 'svelte';

	import CircleAlertIcon from 'lucide-svelte/icons/circle-alert';
	import InfoIcon from 'lucide-svelte/icons/info';
	import TriangleAlertIcon from 'lucide-svelte/icons/triangle-alert';

	import KatexBlock from '../KatexBlock.svelte';

	type $$Props = {
		label: string;
		type: 'info' | 'warning' | 'error';
		children: Snippet;
	};

	let { label, type = 'info', children }: $$Props = $props();

	const borderAndBackgroundClasses = $derived(
		type === 'warning'
			? 'border-yellow-400/60 bg-yellow-100/50'
			: type === 'error'
				? 'border-red-400/60 bg-red-100/50'
				: 'border-neutral-400/60 bg-neutral-100/50'
	);

	const iconColorClass = $derived(
		type === 'warning' ? 'text-yellow-700' : type === 'error' ? 'text-red-700' : 'text-neutral-600'
	);
</script>

<div
	class={`flex flex-col py-0.5 p-1 text-neutral-800 items-start border ${borderAndBackgroundClasses}`}
>
	<span
		class={`font-bold uppercase text-2xs font-mono tracking-wider flex items-center gap-1.25 ${iconColorClass}`}
	>
		{#if type === 'info'}
			<InfoIcon class="w-3 h-3 shrink-0" strokeWidth={2.8} />
		{:else if type === 'warning'}
			<TriangleAlertIcon class="w-3 h-3 shrink-0" strokeWidth={2.8} />
		{:else if type === 'error'}
			<CircleAlertIcon class="w-3 h-3 shrink-0" strokeWidth={2.8} />
		{/if}
		<KatexBlock text={label} />
	</span>
	<span class="text-xs">
		{@render children?.()}
	</span>
</div>
