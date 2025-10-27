<script lang="ts">
	import type { Settings2 } from 'lucide-svelte';
	import type { Snippet } from 'svelte';
	import type { AriaRole } from 'svelte/elements';

	type $$Props = {
		icon: typeof Settings2;
		class?: string;
		role?: AriaRole;
		colorClass?: string;
		strokeWidth?: number;
		children: Snippet;
	};

	let {
		role = 'complementary',
		class: additionalClasses = '',
		children,
		icon: Icon,
		colorClass = 'border-purple-400 bg-purple-100 text-purple-800 w-full'
	}: $$Props = $props();
</script>

<div
	class={`flex items-center gap-2 ${additionalClasses}`.trim()}
	aria-label="User guide tip"
	{role}
	data-nosnippet
>
	<span class={`inline-flex items-start p-2 border text-base font-medium border-l-4 ${colorClass}`}>
		<Icon class="w-5 h-5 mr-2 mt-0.5 flex-shrink-0" aria-hidden="true" strokeWidth={1.8} />
		<div>
			{@render children?.()}
		</div>
	</span>
</div>
