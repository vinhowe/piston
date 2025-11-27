<script lang="ts">
	import { twMerge } from 'tailwind-merge';

	type $$Props = {
		direction: 'down' | 'right';
		size?: number;
		strokeWidth?: number;
		class?: string;
	};

	let { direction, strokeWidth = 3, class: svgClass = '' }: $$Props = $props();
</script>

<svg
	class={twMerge(`w-3.5 h-3.5 transition-transform translate-y-[0.5px]`, svgClass)}
	class:rotate-90={direction === 'down'}
	fill="none"
	stroke="currentColor"
	viewBox="0 0 24 24"
>
	<path stroke-width={strokeWidth} d="M9 5l7 7-7 7"></path>
</svg>
