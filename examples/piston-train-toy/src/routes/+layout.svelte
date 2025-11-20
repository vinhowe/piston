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
	<meta name="color-scheme" content="light" />
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
	<link rel="preconnect" href="https://use.typekit.net" crossorigin="anonymous" />
	<link
		rel="preload"
		href="https://use.typekit.net/ixr7lrv.css"
		as="style"
		crossorigin="anonymous"
	/>
	<link rel="stylesheet" href="https://use.typekit.net/ixr7lrv.css" crossorigin="anonymous" />
	<link
		rel="preload"
		href="/Berkeley Mono Variable.woff2"
		as="font"
		type="font/woff2"
		crossorigin="anonymous"
	/>

	<!-- Open Graph -->
	<meta property="og:title" content="Sequence Toy" />
	<meta property="og:description" content="Train language models in your browser with WebGPU" />
	<meta property="og:type" content="website" />
	<meta property="og:url" content="https://sequence.toys" />
	<meta property="og:image" content="https://sequence.toys/social.png" />
	<meta property="og:image:alt" content="Preview of Sequence Toy interface" />

	<!-- Twitter -->
	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:title" content="Sequence Toy" />
	<meta name="twitter:description" content="Train language models in your browser with WebGPU" />
	<meta name="twitter:image" content="https://sequence.toys/social.png" />
	<meta name="twitter:creator" content="@vinhowe" />
	<meta name="twitter:site" content="@vinhowe" />

	<!-- Structured data -->
	<script type="application/ld+json">
		{
			"@context": "https://schema.org",
			"@type": "WebApplication",
			"name": "Sequence Toy",
			"url": "https://sequence.toys",
			"description": "Train language models in your browser with WebGPU",
			"inLanguage": "en",
			"applicationCategory": "EducationalApplication",
			"operatingSystem": "Any",
			"creator": {
				"@type": "Person",
				"name": "Vin Howe",
				"sameAs": ["https://vin.how", "https://x.com/vinhowe"]
			}
		}
	</script>
</svelte:head>

{@render children()}
