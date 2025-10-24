<script lang="ts">
	import '../app.css';

	let { children } = $props();
</script>

<svelte:head>
	<title>sequence toy</title>
	<meta name="description" content="Train language models in your browser with WebGPU" />
	<meta name="author" content="Vin Howe" />
	<meta name="keywords" content="machine learning, transformer, WebGPU, training, neural network" />
	<meta name="theme-color" content="oklch(0.438 0.218 303.724)" />
	<meta charset="utf-8" />
	<meta name="robots" content="index, follow" />
	<meta name="googlebot" content="index, follow" />
	<link rel="icon" href="/favicon.ico" sizes="32x32" />
	<link rel="icon" href="/icon.svg" type="image/svg+xml" />
	<link rel="apple-touch-icon" href="/apple-touch-icon.png" />
	<meta name="application-name" content="Sequence Toy" />
	<meta name="subject" content="Machine Learning Training Interface" />
	<meta name="rating" content="general" />
	<link rel="canonical" href="https://sequence.toys" />
</svelte:head>

{@render children()}
