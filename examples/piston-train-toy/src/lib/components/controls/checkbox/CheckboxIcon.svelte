<script lang="ts">
	let { checked }: { checked: boolean } = $props();
</script>

<span
	class="h-4 w-4 flex shrink-0 items-center justify-center peer-focus-visible:ring-2 peer-focus-visible:ring-offset-1 peer-focus-visible:ring-blue-500 transition-opacity duration-100"
	class:bg-neutral-600={checked}
	class:bg-neutral-500={!checked}
>
	<svg
		class={`h-3.5 w-3.5 transition-opacity duration-100 ${checked ? 'opacity-100' : 'opacity-0'}`}
		aria-hidden={!checked}
		viewBox="0 0 16 16"
		fill="white"
		xmlns="http://www.w3.org/2000/svg"
	>
		<path
			d="M12.207 4.793a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0l-2-2a1 1 0 011.414-1.414L6.5 9.086l4.293-4.293a1 1 0 011.414 0z"
		/>
	</svg>
</span>
