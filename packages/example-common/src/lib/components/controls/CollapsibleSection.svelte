<script lang="ts">
	import type { Snippet } from 'svelte';

	import { twMerge } from 'tailwind-merge';
	import ChevronIcon from '../ChevronIcon.svelte';

	type $$Props = {
		title: string;
		isOpen: boolean;
		class?: string;
		ontoggle?: () => void;
		contentClass?: string;
		children?: Snippet;
	};

	let {
		title,
		isOpen,
		class: wrapperClass = '',
		ontoggle,
		contentClass = 'p-2',
		children
	}: $$Props = $props();

	function handleClick() {
		if (ontoggle) {
			ontoggle();
		}
	}

	// FOR TAILWIND, DON'T DELETE:
	// bg-neutral-300
</script>

<div class={twMerge(`border-b border-panel-border-base overflow-hidden`, wrapperClass)}>
	<div
		class="px-1 py-0.5 flex justify-between items-center cursor-pointer select-none bg-gradient-to-r from-neutral-100 to-neutral-300 border-panel-border-base"
		class:border-b={isOpen}
		onclick={handleClick}
		role="button"
		tabindex="0"
		aria-expanded={isOpen}
		aria-controls={`section-content-${title.toLowerCase().replace(/\s+/g, '-')}`}
		onkeydown={(e) => {
			if (e.key === 'Enter' || e.key === ' ') {
				e.preventDefault();
				handleClick();
			}
		}}
	>
		<h2 class="text-neutral-800 text-base font-medium">{title}</h2>
		<ChevronIcon direction={isOpen ? 'down' : 'right'} class="text-neutral-700" />
	</div>

	{#if isOpen}
		<div class={contentClass} id={`section-content-${title.toLowerCase().replace(/\s+/g, '-')}`}>
			{@render children?.()}
		</div>
	{/if}
</div>
