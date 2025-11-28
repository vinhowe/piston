<script lang="ts">
	type Variant = 'prompt' | 'correct' | 'incorrect' | 'generated' | 'neutral';

	let {
		actualText,
		targetText = null,
		variant = 'neutral',
		highlighted = false,
		exampleIndex = 0,
		tokenIndex = 0,
		disabled = false,
		onHover,
		onLeave,
		onSelect
	}: {
		actualText: string;
		targetText?: string | null;
		variant?: Variant;
		highlighted?: boolean;
		exampleIndex?: number;
		tokenIndex?: number;
		disabled?: boolean;
		onHover?: (exampleIndex: number, tokenIndex: number) => void;
		onLeave?: () => void;
		onSelect?: (exampleIndex: number, tokenIndex: number) => void;
	} = $props();

	function handleMouseEnter() {
		if (disabled) return;
		onHover?.(exampleIndex, tokenIndex);
	}

	function handleClick(e: Event) {
		if (disabled) return;
		onSelect?.(exampleIndex, tokenIndex);
		e.stopPropagation();
	}

	function handleMouseLeave() {
		if (disabled) return;
		onLeave?.();
	}

	function visualizeToken(text: string): string {
		return text.replace(/\s/g, '␣');
	}

	function boxClasses(kind: 'actual' | 'target'): string {
		// When highlighted, remove borders; container shows a black ring.
		switch (variant) {
			case 'prompt': {
				return 'bg-white text-neutral-800 p-0.5 border border-neutral-200';
			}
			case 'correct': {
				return 'bg-green-200 text-green-800 p-0.5 border border-green-300';
			}
			case 'incorrect': {
				if (kind === 'target') {
					return 'bg-red-200 text-red-800 p-0.5 border border-red-300';
				} else {
					return 'bg-white text-neutral-800 p-0.5 border border-neutral-200';
				}
			}
			case 'generated': {
				return 'bg-purple-200 text-purple-800 p-0.5 border border-purple-300';
			}
			case 'neutral':
			default: {
				return 'bg-white text-neutral-800 p-0.5 border border-neutral-200';
			}
		}
	}

	const containerBase = $derived(
		targetText != null ? 'inline-grid grid-rows-2 gap-[2px]' : 'inline-flex items-center'
	);

	const zClasses = $derived(highlighted ? 'z-20' : 'z-0 hover:z-20');
	const ringClasses = $derived(highlighted ? 'ring-2 ring-blue-300' : '');

	const containerClasses = $derived(
		`relative ${containerBase} leading-none ml-[2px] first:ml-0 group ${zClasses} shrink-0 ${targetText == null || variant === 'incorrect' ? ringClasses : ''} ${disabled ? '' : 'cursor-pointer'}`
	);
</script>

<span
	class={containerClasses}
	role="button"
	tabindex="0"
	onmouseenter={handleMouseEnter}
	onmouseleave={handleMouseLeave}
	onclick={handleClick}
	onkeydown={(e) => {
		if (e.key === 'Enter' || e.key === ' ') {
			e.preventDefault();
			handleClick(e);
		}
	}}
>
	{#if targetText != null}
		<span
			class={`relative ${boxClasses('target')} ${variant !== 'incorrect' ? ringClasses : ''} row-start-1`}
			aria-hidden="true"
		>
			{visualizeToken(targetText)}
		</span>
		<span class="relative">
			{#if actualText}
				<span
					class={`absolute left-0 top-0 ${boxClasses('actual')} ${variant !== 'incorrect' ? 'opacity-0 pointer-events-none' : ''} group-hover:z-10`}
				>
					{visualizeToken(actualText)}
				</span>
			{/if}
			<span class="relative opacity-0 select-none pointer-events-none row-start-1">
				{visualizeToken(targetText)}
			</span>
		</span>
	{:else}
		<span class={`relative ${boxClasses('actual')}`} aria-hidden="true">
			{visualizeToken(actualText)}
		</span>
	{/if}
</span>
