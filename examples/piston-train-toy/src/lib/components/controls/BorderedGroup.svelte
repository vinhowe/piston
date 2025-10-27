<script lang="ts">
	import type { Snippet } from 'svelte';

	import KatexBlock from '../KatexBlock.svelte';
	import Citations, { type CitationEntries as CitationsType } from './Citations.svelte';

	type $$Props = {
		title?: string;
		id?: string;
		citations?: CitationsType;
		class?: string;
		headerClass?: string;
		contentClass?: string;
		header?: Snippet;
		action?: Snippet;
		children?: Snippet;
	};

	let {
		title,
		id,
		citations,
		class: rootClass = '',
		headerClass = 'bg-panel px-1 py-0.5 flex justify-between items-center text-neutral-800 gap-2',
		contentClass,
		header,
		action,
		children
	}: $$Props = $props();

	if (!title && !header) {
		throw new Error('Either title or header must be provided');
	}
</script>

<div {id} class={`border border-panel-border-base overflow-hidden ${rootClass}`.trim()}>
	<div class={headerClass}>
		{#if title}
			<div class="flex flex-col">
				<h3 class="text-base font-medium">
					<KatexBlock text={title} />
				</h3>
				{#if citations}
					<Citations {citations} />
				{/if}
			</div>
		{:else}
			{@render header?.()}
		{/if}
		{#if action}
			{@render action?.()}
		{/if}
	</div>
	{#if contentClass}
		<div class={contentClass}>
			{@render children?.()}
		</div>
	{:else}
		{@render children?.()}
	{/if}
</div>
