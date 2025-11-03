<script lang="ts">
	let { checked }: { checked: boolean } = $props();
</script>

<span
	class="relative h-4 w-4 shrink-0 rounded-full peer-focus-visible:ring-2 peer-focus-visible:ring-offset-1 peer-focus-visible:ring-blue-500"
>
	<svg viewBox="0 0 16 16" class="block h-4 w-4" aria-hidden="true">
		<!-- Outer ring -->
		<circle
			cx="8"
			cy="8"
			r="7"
			stroke-width="1"
			class="transition-colors duration-100"
			class:fill-neutral-600={checked}
			class:fill-neutral-50={!checked}
			class:stroke-neutral-600={checked}
			class:stroke-neutral-400={!checked}
		/>
		<!-- Inner dot -->
		<circle
			cx="8"
			cy="8"
			r="4"
			class="fill-white transition-opacity duration-100"
			class:opacity-100={checked}
			class:opacity-0={!checked}
		/>
	</svg>
</span>
