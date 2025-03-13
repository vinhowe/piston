<script lang="ts">
	const { label, value } = $props<{label: string, value: number}>();
</script>

<div class="form-group">
	<label class="block text-sm font-medium mb-1">{label}</label>
	<div class="relative">
		<slot name="value-display" />
		<div class="relative h-8">
			<div class="absolute top-1 w-full h-1 bg-gray-200 rounded" />
			<slot name="ticks" />
			<slot name="input" />
		</div>
	</div>
</div>

<style>
	:global(.slider-input) {
		position: absolute;
		top: 0.25rem;
		width: 100%;
		height: 0.25rem;
		appearance: none;
		background-color: transparent;
		cursor: pointer;
	}

	:global(.slider-input::-webkit-slider-thumb) {
		-webkit-appearance: none !important;
		width: 8px !important;
		height: 16px !important;
		background: #374151 !important;
		border: none !important;
		border-radius: 0 !important;
		cursor: pointer;
		transition: background-color 150ms;
	}

	:global(.slider-input::-webkit-slider-thumb:hover) {
		background: #4B5563 !important;
	}

	:global(.slider-input::-moz-range-thumb) {
		width: 8px !important;
		height: 16px !important;
		background: #374151 !important;
		border: none !important;
		border-radius: 0 !important;
		cursor: pointer;
		transition: background-color 150ms;
	}

	:global(.slider-input::-moz-range-thumb:hover) {
		background: #4B5563 !important;
	}

	:global(.slider-tick) {
		position: absolute;
		transform: translateX(-50%);
	}

	:global(.slider-tick-mark) {
		margin-left: auto;
		margin-right: auto;
	}

	:global(.slider-tick-mark.major) {
		height: 0.5rem;
		width: 0.125rem;
		background-color: rgb(156, 163, 175);
	}

	:global(.slider-tick-mark.medium) {
		height: 0.375rem;
		width: 0.125rem;
		background-color: rgb(209, 213, 219);
	}

	:global(.slider-tick-mark.minor) {
		height: 0.25rem;
		width: 0.125rem;
		background-color: rgb(229, 231, 235);
	}

	:global(.slider-tick-mark.snap) {
		background-color: rgb(156, 163, 175);
	}

	:global(.slider-tick-label) {
		font-size: 0.675rem;
		color: rgb(156, 163, 175);
		margin-top: 0.125rem;
		font-family: ui-monospace, monospace;
		user-select: none;
	}
</style> 