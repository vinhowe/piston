<script lang="ts">
	import type { Snippet } from 'svelte';

	import ChevronIcon from '../ChevronIcon.svelte';

	type $$Props = {
		title: string;
		groupName?: string;
		isOpen: boolean;
		onToggle?: () => void;
		class?: string;
		contentClass?: string;
		enabled?: boolean;
		chips?: Snippet;
		children?: Snippet;
	};

	let {
		title,
		groupName,
		isOpen,
		onToggle,
		class: wrapperClass = '',
		contentClass = 'grid grid-cols-1 @3xl:grid-cols-2 @6xl:grid-cols-3 gap-4',
		enabled = true,
		chips,
		children
	}: $$Props = $props();

	function handleClick() {
		if (onToggle) onToggle();
	}
</script>

<div class={`space-y-3 ${wrapperClass}`.trim()}>
	<div class="flex items-start gap-3">
		<div
			class="flex items-center gap-2 select-none cursor-pointer"
			role="button"
			tabindex="0"
			aria-expanded={isOpen}
			aria-controls={groupName ? `metrics-section-${groupName}` : undefined}
			onclick={onToggle}
			onkeydown={(e) => {
				if (e.key === 'Enter' || e.key === ' ') {
					e.preventDefault();
					handleClick();
				}
			}}
		>
			<ChevronIcon
				direction={isOpen ? 'down' : 'right'}
				class={enabled ? 'text-neutral-700' : 'text-neutral-400'}
			/>
			<h3 class="text-lg font-medium text-neutral-800 capitalize">{title}</h3>
		</div>
		{@render chips?.()}
	</div>

	{#if isOpen}
		<div class={contentClass} id={groupName ? `metrics-section-${groupName}` : undefined}>
			{@render children?.()}
		</div>
	{/if}
</div>
