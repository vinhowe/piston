<script lang="ts">
	import type { Snippet } from 'svelte';

	import CheckboxInput from './checkbox/CheckboxInput.svelte';
	import BorderedGroup from './BorderedGroup.svelte';
	import type { CitationEntries as CitationsType } from './Citations.svelte';

	type $$Props = {
		title?: string;
		id: string;
		citations?: CitationsType;
		showEnableToggle?: boolean;
		enabled?: boolean;
		class?: string;
		headerClass?: string;
		contentClass?: string;
		header?: Snippet;
		children?: Snippet;
		hasDefaultValue?: boolean;
		onReset?: () => void;
	};

	let {
		title,
		id,
		citations,
		showEnableToggle = false,
		enabled = $bindable(false),
		class: rootClass = '',
		headerClass = undefined,
		contentClass = 'p-1 space-y-1',
		header,
		children,
		hasDefaultValue = false,
		onReset = undefined
	}: $$Props = $props();
</script>

<BorderedGroup {title} {id} {citations} {header} class={rootClass} {headerClass}>
	{#snippet action()}
		{#if showEnableToggle}
			<CheckboxInput
				id={`${id}-enable-toggle`}
				bind:checked={enabled}
				{hasDefaultValue}
				{onReset}
			/>
		{/if}
	{/snippet}
	{#if !showEnableToggle || (showEnableToggle && enabled)}
		<div class={contentClass}>
			{@render children?.()}
		</div>
	{/if}
</BorderedGroup>
