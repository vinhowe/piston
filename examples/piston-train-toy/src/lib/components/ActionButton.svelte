<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		color?: 'blue' | 'red' | 'green' | 'yellow' | 'purple' | 'gray';
		colorClass?: string;
		class?: string;
		disabled?: boolean;
		children?: Snippet;
		[key: string]: unknown;
	}

	let {
		color,
		colorClass,
		class: additionalClasses,
		disabled,
		children,
		...restProps
	}: Props = $props();

	const COLOR_MAP = {
		blue: 'text-blue-900 bg-gradient-to-t from-blue-300 to-blue-100 border-blue-600',
		red: 'text-red-900 bg-gradient-to-t from-red-300 to-red-100 border-red-600',
		green: 'text-green-900 bg-gradient-to-t from-green-300 to-green-100 border-green-600',
		yellow: 'text-yellow-900 bg-gradient-to-t from-yellow-300 to-yellow-100 border-yellow-600',
		purple: 'text-purple-900 bg-gradient-to-t from-purple-300 to-purple-100 border-purple-600',
		gray: 'text-neutral-900 bg-gradient-to-t from-neutral-300 to-neutral-100 border-neutral-600'
	};

	const computedColorClass = $derived(
		color ? COLOR_MAP[color] : colorClass || 'bg-blue-700 shadow-blue-500/50'
	);

	const baseClasses =
		'border cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 disabled:saturate-50 py-1 px-2 text-sm font-mono uppercase tracking-wide font-medium';
</script>

<button
	type="button"
	class="{baseClasses} {computedColorClass} {additionalClasses || ''}"
	{disabled}
	{...restProps}
>
	{@render children?.()}
</button>
