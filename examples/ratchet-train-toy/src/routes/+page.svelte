<script lang="ts">
	import { onMount } from 'svelte';
	import TrainWorker from '../trainWorker.ts?worker';
	import { Chart, type ChartItem } from 'chart.js/auto';
	import Slider from '$lib/components/Slider.svelte';
	import LogSlider from '$lib/components/LogSlider.svelte';
	import TickSlider from '$lib/components/TickSlider.svelte';
	import ActivationPicker from '$lib/components/ActivationPicker.svelte';
	import { taskMetadata } from '../tasks';

	let loss = 0;
	let evalAccuracy = 0;
	let initialTotalLoss: number | null = null;
	let worker: Worker;
	let chart: Chart;
	let speedChart: Chart;
	let memoryChart: Chart;
	let lrChart: Chart;
	let accuracyChart: Chart;
	let canvas: HTMLCanvasElement;
	let speedCanvas: HTMLCanvasElement;
	let memoryCanvas: HTMLCanvasElement;
	let lrCanvas: HTMLCanvasElement;
	let accuracyCanvas: HTMLCanvasElement;
	let attentionCanvases: HTMLCanvasElement[][] = [];
	let showAdvanced = false;
	let runCount = 0;
	let runConfigs: Record<string, any>[] = [];
	let lastStepTime: number;
	let messages: { text: string; timestamp: string }[] = [];
	let currentStepCount = 0;
	let isTraining = false;
	let currentModelLayers = 0;
	let currentModelHeads = 0;
	let showAttention = true;
	let attentionOnly = false;
	let positional_encoding = 'rope';
	let seed: string | undefined = 'autoregressive';
	let layernorm_position = 'pre';
	let hasWebGPU = true;

	// Model parameters
	let n_layer = 1;
	let n_head = 6;
	let n_8_embd_per_head = 2;
	let n_embd = n_8_embd_per_head * 8 * n_head;
	let batch_size = 2;
	let dataset = 'sort';
	let activation = 'gelu';

	// Optimizer parameters
	let lr = -3; // log10 scale
	let beta1 = 0.9;
	let beta2 = 0.999;
	let eps = -8; // log10 scale
	let weight_decay = -1; // log10 scale

	// Track last parameter values to detect actual changes
	let lastParams: Record<string, any> | null = null;
	let lastTaskParams: Record<string, number> | null = null;

	// Add reactive statement to restart training when parameters change
	$: {
		const currentParams = {
			n_layer,
			n_head,
			n_embd,
			batch_size,
			dataset,
			activation,
			attentionOnly,
			positional_encoding,
			seed,
			lr,
			beta1,
			beta2,
			eps,
			weight_decay,
			optimizer_type,
			scheduler_type,
			momentum,
			scheduler_factor,
			scheduler_steps,
			scheduler_eta_min,
			layernorm_position
		};

		if (lastParams === null) {
			// First time through, just store the parameters
			lastParams = { ...currentParams };
			lastTaskParams = { ...taskParameters };
		} else {
			// Check if any parameter actually changed
			const hasChanges = Object.entries(currentParams).some(
				([key, value]) => lastParams![key] !== value
			);

			// Check which task parameters changed
			const hasTaskChanges =
				Object.keys(lastTaskParams!).length > 0 &&
				Object.entries(taskParameters).some(([key, value]) => lastTaskParams![key] !== value);

			lastTaskParams = { ...taskParameters };

			if (worker && isTraining && (hasChanges || hasTaskChanges)) {
				// Update last params before restarting
				lastParams = { ...currentParams };
				// Restart training with new parameters
				stopTraining();
				startTraining();
			}
		}
	}

	const colors = [
		'rgb(75, 192, 192)',
		'rgb(153, 102, 255)',
		'rgb(255, 99, 132)',
		'rgb(255, 205, 86)',
		'rgb(54, 162, 235)'
	];

	// $: n_embd = Math.ceil(n_embd / n_head) * n_head; // Ensure n_embd is multiple of n_head
	$: n_embd = n_8_embd_per_head * 8 * n_head;

	// Add new state variables
	let optimizer_type = 'adamw';
	let scheduler_type = 'none';
	let momentum = 0.0;
	let scheduler_factor = 1.5;
	let scheduler_steps = 1000;
	let scheduler_eta_min = 1e-4;

	// Add new state for batch visualization
	let batchHistory: Array<{
		input: string[];
		target: string[];
		losses: number[];
		initialLoss: number;
	}> = [];
	const MAX_HISTORY = 5; // Keep last 10 batches

	// Add eval history state
	let evalHistory: Array<{
		sequence: number[];
		completion: number[];
		target: number[];
		results: Array<boolean | null>;
		logits: number[];
	}> = [];
	const MAX_EVAL_HISTORY = 5;

	// Add task parameter state
	let taskParameters: Record<string, number> = {};

	// Initialize task parameters with defaults
	$: {
		if (dataset && taskMetadata[dataset]) {
			const metadata = taskMetadata[dataset];
			for (const [key, param] of Object.entries(metadata.parameters)) {
				if (!(key in taskParameters)) {
					taskParameters[key] = param.default;
				}
			}
		}
	}

	function initializeAttentionCanvases(numLayers: number, numHeads: number) {
		if (typeof document === 'undefined') return; // Check if we're in the browser

		attentionCanvases = Array(numLayers)
			.fill(null)
			.map(() =>
				Array(numHeads)
					.fill(null)
					.map(() => {
						const canvas = document.createElement('canvas');
						const size = 200;
						canvas.width = size;
						canvas.height = size;
						return canvas;
					})
			);
	}

	function getChangedParams() {
		if (runConfigs.length <= 1) return new Set<string>();

		const paramSets = new Map<string, Set<any>>();
		const allParams = [
			'n_layer',
			'n_head',
			'n_embd',
			'batch_size',
			'dataset',
			'activation',
			'attention_only',
			'positional_encoding',
			'lr',
			'beta1',
			'beta2',
			'eps',
			'weight_decay',
			'layernorm_position'
		];

		for (const config of runConfigs) {
			for (const param of allParams) {
				if (!paramSets.has(param)) {
					paramSets.set(param, new Set());
				}
				const value =
					param === 'lr'
						? config.optimizer.lr
						: param.startsWith('beta') || param === 'eps' || param === 'weight_decay'
							? config.optimizer[param]
							: config[param];
				paramSets.get(param)!.add(JSON.stringify(value));
			}
		}

		return new Set(
			Array.from(paramSets.entries())
				.filter(([_, values]) => values.size > 1)
				.map(([param]) => param)
		);
	}

	function updateAllLabels() {
		const changedParams = getChangedParams();
		chart.data.datasets.forEach((dataset, index) => {
			const config = runConfigs[index];
			dataset.label = formatRunLabel(config, index + 1);
		});
		chart.update();
	}

	function formatRunLabel(config: Record<string, any>, runNumber: number) {
		const changedParams = getChangedParams();
		const parts = [`#${runNumber}`];

		// Only add parameter section if there are changes
		if (changedParams.size > 0) {
			const paramParts = Array.from(changedParams).map((param) => {
				const value =
					param === 'lr'
						? config.optimizer.lr.toExponential(1)
						: param.startsWith('beta') || param === 'eps' || param === 'weight_decay'
							? config.optimizer[param]
							: config[param];
				return `${param}=${value}`;
			});
			if (paramParts.length > 0) {
				parts.push(`[${paramParts.join(', ')}]`);
			}
		}

		return parts.join(' ');
	}

	function addMessage(text: string) {
		const now = new Date();
		const timestamp = now.toLocaleTimeString('en-US', { hour12: false });
		messages = [...messages, { text, timestamp }];
	}

	function stringToSeed(input: string | undefined): BigInt | undefined {
		if (!input) return undefined;

		// Simple string hash function that produces a number
		let hash = 0;
		for (let i = 0; i < input.length; i++) {
			const char = input.charCodeAt(i);
			hash = (hash << 5) - hash + char;
			hash = hash >>> 0; // Convert to 32-bit unsigned integer
		}

		// Convert to BigInt for 64-bit operations
		const hash64 = BigInt(hash) * BigInt(2654435761); // Multiply by prime for better distribution

		// Return as number, which is fine since we just need a seed
		return BigInt(hash64 & BigInt('0xFFFFFFFFFFFFFFFF'));
	}

	function startTraining() {
		currentStepCount = 0;
		// Clear evaluation and batch histories
		evalHistory = [];
		batchHistory = [];

		// isTraining = true;
		// Store the current model configuration
		currentModelLayers = n_layer;
		currentModelHeads = n_head;
		// Initialize canvases for the new configuration
		initializeAttentionCanvases(currentModelLayers, currentModelHeads);

		addMessage(`Initializing model (run #${runCount + 1})...`);
		const config = {
			vocab_size: 256,
			n_embd,
			n_layer,
			n_head,
			block_size: 24,
			batch_size,
			dataset,
			task_parameters: taskParameters,
			activation,
			attention_only: attentionOnly,
			positional_encoding,
			seed: stringToSeed(seed),
			optimizer: {
				optimizer_type,
				lr: Math.pow(10, lr),
				beta1,
				beta2,
				eps: Math.pow(10, eps),
				weight_decay: Math.pow(10, weight_decay),
				momentum,
				scheduler_type,
				scheduler_factor,
				scheduler_steps,
				scheduler_eta_min
			},
			layernorm_position
		};

		// Log model parameters
		addMessage(`Model parameters:
  - Embedding dim: ${n_embd}
  - Layers: ${n_layer}
  - Heads: ${n_head}
  - Batch size: ${batch_size}
  - Dataset: ${dataset}
  - Activation: ${activation}
  - Positional encoding: ${positional_encoding.charAt(0).toUpperCase() + positional_encoding.slice(1)}
  - Learning rate: ${Math.pow(10, lr).toExponential(1)}
  - Beta1: ${beta1}
  - Beta2: ${beta2}
  - Epsilon: ${Math.pow(10, eps).toExponential(1)}
  - Weight decay: ${Math.pow(10, weight_decay).toExponential(1)}`);

		runConfigs.push(config);
		runCount++;

		// Add dataset to loss chart
		chart.data.datasets.push({
			label: '', // Will be set by updateAllLabels
			data: [],
			borderColor: colors[runCount % colors.length],
			tension: 0.1,
			order: -runCount, // Newer runs (higher runCount) will be drawn on top
			pointStyle: false
		});

		// Add dataset to speed chart
		speedChart.data.datasets.push({
			label: '', // Will be set by updateAllLabels
			data: [],
			borderColor: colors[runCount % colors.length],
			tension: 0.1,
			order: -runCount,
			pointStyle: false
		});

		// Add dataset to memory chart
		memoryChart.data.datasets.push({
			label: '', // Will be set by updateAllLabels
			data: [],
			borderColor: colors[runCount % colors.length],
			tension: 0.1,
			order: -runCount,
			pointStyle: false
		});

		// Add dataset to learning rate chart
		lrChart.data.datasets.push({
			label: '', // Will be set by updateAllLabels
			data: [],
			borderColor: colors[runCount % colors.length],
			tension: 0.1,
			order: -runCount,
			pointStyle: false
		});

		// Add dataset to accuracy chart
		accuracyChart.data.datasets.push({
			label: '', // Will be set by updateAllLabels
			data: [],
			borderColor: colors[runCount % colors.length],
			tension: 0.1,
			order: -runCount,
			pointStyle: false
		});

		updateAllLabels();
		lastStepTime = performance.now();

		// Send the new config to the existing worker
		worker.postMessage(config);
	}

	function stopTraining() {
		worker.postMessage({ type: 'stop' });
		isTraining = false;
		attentionCanvases = []; // Clear canvases when stopping
	}

	// function clearChart() {
	// 	chart.data.datasets = [];
	// 	speedChart.data.datasets = [];
	// 	memoryChart.data.datasets = [];
	// 	runCount = 0;
	// 	runConfigs = [];
	// 	chart.update();
	// 	speedChart.update();
	// 	memoryChart.update();
	// 	startTraining();
	// }

	onMount(async () => {
		// Check for WebGPU support
		hasWebGPU = 'gpu' in navigator;

		// Remove the canvas initialization since it's now handled by the reactive statement

		// Initialize worker once
		worker = new TrainWorker();
		addMessage('Starting worker...');
		worker.onmessage = (e: MessageEvent) => {
			const data = e.data;

			switch (data.type) {
				case 'ready':
					addMessage('Worker ready');
					startTraining(); // Start initial training run
					break;
				case 'modelReady':
					addMessage('Model initialized and ready');
					break;
				case 'evalStreaming':
					// Convert sequence and completion to strings
					const sequence = data.sequence;
					const completion = data.completion;
					const target = data.target;

					// Update eval history - find existing entry or create new one
					const existingIndex = evalHistory.findIndex((entry) =>
						entry.sequence.every((x, i) => x === sequence[i])
					);
					if (existingIndex !== -1) {
						// Update existing entry
						evalHistory[existingIndex] = {
							...evalHistory[existingIndex],
							completion,
							target,
							results: data.evalResult,
							logits: data.logits.data
						};
						evalHistory = evalHistory; // Trigger reactivity
					} else {
						// Create new entry
						evalHistory = [
							{
								sequence,
								completion,
								target,
								results: data.evalResult,
								logits: data.logits.data
							},
							...evalHistory.slice(0, MAX_EVAL_HISTORY - 1)
						];
					}
					break;
				case 'step':
					currentStepCount++;
					if (!isTraining) {
						isTraining = true;
					}

					if (currentStepCount === 3) {
						addMessage('Graph hash is consistent; reallocating with shared objects...');
					}

					const currentTime = performance.now();
					const stepsPerSecond = 1000 / (currentTime - lastStepTime);
					lastStepTime = currentTime;

					loss = data.loss.total;
					if (initialTotalLoss === null) {
						initialTotalLoss = loss;
					}

					// Process batch data for visualization
					const tokenLosses = data.loss.tokens.map((x: number) => x * data.loss.tokens.length);

					// Update batch history
					batchHistory = [
						{
							input: data.input,
							target: data.target,
							losses: tokenLosses,
							initialLoss: initialTotalLoss / 2
						},
						...batchHistory.slice(0, MAX_HISTORY - 1)
					];

					const currentLossDataset = chart.data.datasets[chart.data.datasets.length - 1];
					const currentSpeedDataset = speedChart.data.datasets[speedChart.data.datasets.length - 1];
					const currentMemoryDataset =
						memoryChart.data.datasets[memoryChart.data.datasets.length - 1];
					const currentLrDataset = lrChart.data.datasets[lrChart.data.datasets.length - 1];

					currentLossDataset.data.push(loss);
					currentSpeedDataset.data.push(stepsPerSecond);
					currentMemoryDataset.data.push(Number(data.usage_bytes) / (1024 * 1024)); // Convert bigint to number then to MB
					currentLrDataset.data.push(data.learning_rate);

					// Update accuracy chart only when we have valid accuracy data
					if (data.accuracy !== null && data.accuracy !== undefined) {
						const currentAccuracyDataset =
							accuracyChart.data.datasets[accuracyChart.data.datasets.length - 1];
						evalAccuracy = data.accuracy;
						// Store both the step number and accuracy value
						currentAccuracyDataset.data.push({
							x: currentStepCount,
							y: data.accuracy
						});
						accuracyChart.update();
					}

					// Update labels if this is the longest run
					const maxLength = Math.max(...chart.data.datasets.map((d) => d.data.length));
					const labels = Array.from({ length: maxLength }, (_, i) => i.toString());
					chart.data.labels = labels;
					speedChart.data.labels = labels;
					memoryChart.data.labels = labels;
					lrChart.data.labels = labels;
					accuracyChart.data.labels = labels;

					if (isNaN(loss)) {
						addMessage('Loss is NaN; stopped training');
						worker.postMessage({ type: 'stop' });
						isTraining = false;
					}

					// Only process attention data if visualization is enabled
					if (showAttention) {
						// Store attention masks for visualization
						const attnMasks = new Float32Array(data.attn_masks.data);
						const attnMasksShape = data.attn_masks.shape;
						const seqLen = Math.sqrt(attnMasksShape[3]); // 23 (sqrt(529))

						// Reshape attention masks into [n_layer, n_head, seqLen, seqLen]
						for (let layer = 0; layer < n_layer; layer++) {
							for (let head = 0; head < n_head; head++) {
								const startIdx = layer * n_head * attnMasksShape[3] + head * attnMasksShape[3];
								const attentionMap = attnMasks.slice(startIdx, startIdx + attnMasksShape[3]);

								// Get the canvas context and draw the attention pattern
								const ctx = attentionCanvases[layer][head].getContext('2d')!;
								const imageData = ctx.createImageData(seqLen, seqLen);

								// Convert attention values to grayscale
								for (let i = 0; i < attentionMap.length; i++) {
									const value = Math.floor(attentionMap[attentionMap.length - 1 - i] * 255);
									const idx = i * 4;
									imageData.data[idx] = value; // R
									imageData.data[idx + 1] = 0; // G
									imageData.data[idx + 2] = 0; // B
									imageData.data[idx + 3] = 255; // A
								}

								// Scale up the image to the canvas size
								ctx.putImageData(imageData, 0, 0);
								const tempCanvas = document.createElement('canvas');
								tempCanvas.width = seqLen;
								tempCanvas.height = seqLen;
								tempCanvas.getContext('2d')!.putImageData(imageData, 0, 0);

								// Clear and scale
								ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
								ctx.imageSmoothingEnabled = false;
								ctx.drawImage(
									tempCanvas,
									0,
									0,
									seqLen,
									seqLen,
									0,
									0,
									ctx.canvas.width,
									ctx.canvas.height
								);
							}
						}
					}

					chart.update();
					speedChart.update();
					memoryChart.update();
					lrChart.update();
					accuracyChart.update();
					break;
				case 'error':
					addMessage(`Error: ${data.error}`);
					console.error('Worker error:', data.error);
					isTraining = false;
					break;
			}
		};

		// Initialize charts
		chart = new Chart(canvas, {
			type: 'line',
			data: {
				labels: [],
				datasets: []
			},
			options: {
				responsive: true,
				animation: false,
				scales: {
					y: {
						beginAtZero: true
					}
				},
				plugins: {
					legend: {
						reverse: true
					}
					// title: {
					// 	display: false,
					// 	// text: 'Training Loss',
					// 	font: {
					// 		weight: 'bold'
					// 	}
					// }
				},
				datasets: {
					line: {
						order: -1
					}
				}
			}
		});

		// Initialize speed chart
		speedChart = new Chart(speedCanvas, {
			type: 'line',
			data: {
				labels: [],
				datasets: []
			},
			options: {
				responsive: true,
				animation: false,
				aspectRatio: 4,
				scales: {
					y: {
						beginAtZero: true
					}
				},
				plugins: {
					legend: {
						display: false
					},
					title: {
						display: true,
						text: 'Steps/Second',
						font: {
							weight: 'bold'
						}
					}
				},
				datasets: {
					line: {
						order: -1
					}
				}
			}
		});

		// Initialize memory chart
		memoryChart = new Chart(memoryCanvas, {
			type: 'line',
			data: {
				labels: [],
				datasets: []
			},
			options: {
				responsive: true,
				animation: false,
				aspectRatio: 4,
				plugins: {
					legend: {
						display: false
					},
					title: {
						display: true,
						text: 'GPU Memory Usage (MB)',
						font: {
							weight: 'bold'
						}
					}
				},
				datasets: {
					line: {
						order: -1
					}
				}
			}
		});

		// Initialize learning rate chart
		lrChart = new Chart(lrCanvas, {
			type: 'line',
			data: {
				labels: [],
				datasets: []
			},
			options: {
				responsive: true,
				animation: false,
				aspectRatio: 4,
				scales: {
					y: {
						beginAtZero: true
					}
				},
				plugins: {
					legend: {
						display: false
					},
					title: {
						display: true,
						text: 'Learning Rate',
						font: {
							weight: 'bold'
						}
					}
				},
				datasets: {
					line: {
						order: -1
					}
				}
			}
		});

		// Initialize accuracy chart
		accuracyChart = new Chart(accuracyCanvas, {
			type: 'line',
			data: {
				labels: [],
				datasets: []
			},
			options: {
				responsive: true,
				animation: false,
				aspectRatio: 4,
				scales: {
					x: {
						type: 'linear',
						display: true,
						title: {
							display: true,
							text: 'Steps'
						}
					},
					y: {
						beginAtZero: true,
						max: 1,
						title: {
							display: true,
							text: 'Accuracy'
						}
					}
				},
				plugins: {
					legend: {
						display: false
					},
					title: {
						display: true,
						text: 'Validation Accuracy',
						font: {
							weight: 'bold'
						}
					}
				},
				datasets: {
					line: {
						order: -1,
						showLine: true, // Connect points with lines
						spanGaps: true // Connect points across gaps
					}
				}
			}
		});
	});
</script>

<main class="max-w-5xl mx-auto p-8">
	{#if !hasWebGPU}
		<div class="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 mb-8" role="alert">
			<div class="mb-2">
				<p><b>Toy Transformer requires WebGPU support.</b> Please either:</p>
			</div>
			<ul class="list-disc list-inside">
				<li>
					Use <b>the latest version of Chrome</b>, a
					<b>Chromium-based browser (Edge, Arc, etc.)</b>, or <b>Firefox Nightly</b>.
				</li>
				<li>
					Or, if you're using Safari:
					<ul class="list-disc list-inside ml-4">
						<li>
							MacOS: <a
								href="https://developer.apple.com/documentation/safari-developer-tools/feature-flag-settings"
								class="text-blue-500">Develop menu</a
							> > Feature Flags > WebGPU
						</li>
						<li>iOS: System Settings > Apps > Safari > Advanced > Feature Flags > WebGPU</li>
					</ul>
				</li>
			</ul>
		</div>
	{/if}
	<div class="grid grid-cols-1 md:grid-cols-3 gap-8">
		<div class="space-y-4 col-span-2">
			<div class="flex justify-between items-baseline gap-4 space-y-3">
				<h1 class="text-2xl">toy transformer</h1>
				<div class="flex flex-wrap gap-3 justify-end text-sm md:text-base">
					<a href="https://x.com/vinhowe" class="text-blue-500">@vinhowe</a>
					<a href="https://x.com/grantpitt0" class="text-blue-500">@grantpitt</a>
					<a href="https://github.com/vinhowe/ratchet-backward" class="text-blue-500"
						>github</a
					>
				</div>
			</div>
			<div class="flex gap-6 md:gap-8">
				<p class="font-mono text-sm">train loss: {loss.toFixed(4)}</p>
				<p class="font-mono text-sm text-right">val accuracy: {evalAccuracy.toFixed(4)}</p>
			</div>
			<div class="relative w-full">
				<canvas bind:this={canvas} />
			</div>
			<div class="relative w-full">
				<canvas bind:this={accuracyCanvas} />
			</div>

			<!-- Add checkbox before the attention visualization section -->
			{#if isTraining}
				<div class="bg-gray-100 p-4 flex flex-col gap-4">
					<div class="flex items-center gap-2">
						<input
							type="checkbox"
							id="showAttention"
							bind:checked={showAttention}
							class="w-4 h-4 text-black border-gray-300 rounded focus:ring-0"
						/>
						<label for="showAttention" class="text-sm font-medium">Show Attention</label>
					</div>

					{#if showAttention && attentionCanvases.length > 0}
						<!-- <h2 class="text-lg font-semibold mb-4">Attention Patterns</h2> -->
						<div class="flex flex-col gap-4">
							{#each attentionCanvases as layerCanvases, layerIdx}
								<div class="flex flex-col items-center border border-dashed border-gray-400 p-2">
									<h3 class="text-sm font-medium mb-2">Layer {layerIdx + 1}</h3>
									<div class="flex flex-wrap gap-2 p-0 justify-center">
										{#each layerCanvases as canvas, headIdx}
											<div class="relative">
												<canvas
													bind:this={attentionCanvases[layerIdx][headIdx]}
													class="w-24 aspect-square"
													style="image-rendering: pixelated;"
												></canvas>
												<div class="text-xs text-center mt-1">Head {headIdx + 1}</div>
											</div>
										{/each}
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			{/if}

			{#if isTraining}
				<div class="bg-gray-100 p-4 flex flex-col gap-4">
					<div class="flex items-center justify-between">
						<h3 class="text-sm font-medium">Train Batches</h3>
						<div class="text-xs text-gray-500">Showing last {MAX_HISTORY} batches</div>
					</div>
					<div class="flex flex-col gap-2">
						{#each batchHistory as batch}
							<div class="flex flex-col leading-none border-l-2 border-gray-300 pl-1">
								{#each batch.input as input, batchIdx}
									<div class="flex items-center">
										<div class="font-mono text-sm">
											<span>{input[0]}</span
											>{#each [...input.slice(1), batch.target[batchIdx][batch.target[batchIdx].length - 1]] as char, i}
												<span
													style="background-color: rgba(255, 0, 0, {Math.min(
														1,
														Math.max(
															0,
															batch.losses[batchIdx * input.length + i] / (batch.initialLoss * 2)
														)
													)})"
												>
													{char}
												</span>
											{/each}
										</div>
									</div>
								{/each}
							</div>
						{/each}
					</div>
				</div>

				{#if evalHistory.length > 0}
					<div class="bg-gray-100 p-4 flex flex-col gap-4">
						<div class="flex items-center justify-between">
							<h3 class="text-sm font-medium">Eval Results</h3>
							<div class="text-xs text-gray-500">Showing last {MAX_EVAL_HISTORY} evaluations</div>
						</div>
						<div class="flex flex-col gap-2">
							{#each evalHistory as evalEntry}
								<div class="flex flex-col leading-none border-l-2 border-gray-300 pl-1">
									<div class="flex items-center">
										<div class="font-mono text-sm">
											<span>{evalEntry.sequence.join('')}</span
											>{#each evalEntry.completion as char, i}
												<span
													style="background-color: {evalEntry.results[i] === true
														? 'rgba(0, 255, 0, 0.3)'
														: evalEntry.results[i] === false
															? 'rgba(255, 0, 0, 0.3)'
															: 'transparent'}"
												>
													{char}
												</span>
											{/each}
											<span>&rarr; {evalEntry.target.join('')}</span>
										</div>
									</div>
								</div>
							{/each}
						</div>
					</div>
				{/if}
			{/if}
			<div class="relative w-full">
				<canvas bind:this={lrCanvas} />
			</div>
			<div class="relative w-full">
				<canvas bind:this={speedCanvas} />
			</div>
			<div class="relative w-full">
				<canvas bind:this={memoryCanvas} />
			</div>

			<div
				class="bg-gray-900 text-gray-100 font-mono p-4 text-sm h-64 overflow-y-auto flex flex-col-reverse"
			>
				<div>
					{#each messages as message}
						<div class="leading-5 min-h-[20px] break-words whitespace-pre-wrap">
							<span class="text-gray-500 shrink-0 inline-block">[{message.timestamp}]</span>
							{message.text}
						</div>
					{/each}
				</div>
			</div>
		</div>
		<div class="space-y-3 md:order-first">
			<div class="flex gap-4">
				<!-- <button
					on:click={startTraining}
					class="flex-1 bg-black text-white py-2 px-4 hover:bg-gray-800 transition-colors"
				>
					Train
				</button> -->
				{#if isTraining}
					<button
						on:click={stopTraining}
						class="flex-1 bg-gray-500 text-white py-2 px-4 hover:bg-gray-600 transition-colors"
					>
						Stop
					</button>
				{/if}
			</div>
			<div class="bg-gray-100 p-4">
				<h2 class="text-lg font-semibold mb-4">Dataset</h2>
				<div class="form-group">
					<label class="block text-sm font-medium mb-1">Task</label>
					<div class="relative">
						<select
							bind:value={dataset}
							class="w-full p-2 pr-12 border focus:outline-none focus:border-gray-400 border-gray-400 bg-white appearance-none"
						>
							{#each Object.entries(taskMetadata) as [key, meta]}
								<option value={key}>{meta.name}</option>
							{/each}
						</select>
						<div
							class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400"
						>
							<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"
								><path
									d="M19 9l-7 7-7-7"
									stroke-linecap="round"
									stroke-linejoin="round"
									stroke-width="2"
								/></svg
							>
						</div>
					</div>
					{#if dataset && taskMetadata[dataset]}
						<p class="text-sm text-gray-600 mt-1">{taskMetadata[dataset].description}</p>
						<div class="mt-4 space-y-4">
							{#each Object.entries(taskMetadata[dataset].parameters) as [key, param]}
								<div class="form-group">
									<TickSlider
										bind:value={taskParameters[key]}
										min={param.min}
										max={param.max}
										label={param.name}
									/>
									<p class="text-sm text-gray-600 mt-1">{param.description}</p>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			</div>
			<div class="bg-gray-100 p-4">
				<h2 class="text-lg font-semibold mb-4">Model Architecture</h2>
				<TickSlider bind:value={n_layer} min={1} max={6} label="Number of Transformer Layers" />

				<TickSlider bind:value={n_head} min={1} max={6} label="Number of Attention Heads" />

				<div>
					<TickSlider
						bind:value={n_8_embd_per_head}
						min={1}
						max={8}
						label="Embedding Size"
						formatter={(v) => `${v * 8} per head`}
					/>
				</div>

				<div class="form-group">
					<label class="block text-sm font-medium mb-1">Activation Function</label>
					<ActivationPicker bind:value={activation} />
				</div>

				<div class="form-group mt-2">
					<label class="block text-sm font-medium mb-1">Positional Encoding</label>
					<div class="relative">
						<select
							bind:value={positional_encoding}
							class="w-full p-2 pr-8 border focus:outline-none focus:border-gray-400 border-gray-400 bg-white appearance-none"
						>
							<option value="rope">RoPE</option>
							<option value="alibi">ALiBi</option>
							<option value="learned">Learned</option>
							<option value="sinusoidal">Sinusoidal</option>
						</select>
						<div
							class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400"
						>
							<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path
									d="M19 9l-7 7-7-7"
									stroke-linecap="round"
									stroke-linejoin="round"
									stroke-width="2"
								/>
							</svg>
						</div>
					</div>
				</div>

				<div class="form-group mt-2">
					<label class="block text-sm font-medium mb-1">LayerNorm Position</label>
					<div class="relative">
						<select
							bind:value={layernorm_position}
							class="w-full p-2 pr-8 border focus:outline-none focus:border-gray-400 border-gray-400 bg-white appearance-none"
						>
							<option value="pre">Pre LayerNorm</option>
							<option value="post">Post LayerNorm</option>
						</select>
						<div
							class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400"
						>
							<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path
									d="M19 9l-7 7-7-7"
									stroke-linecap="round"
									stroke-linejoin="round"
									stroke-width="2"
								/>
							</svg>
						</div>
					</div>
				</div>

				<div class="form-group mt-3">
					<div class="flex items-center gap-2">
						<input
							type="checkbox"
							id="attentionOnly"
							bind:checked={attentionOnly}
							class="w-4 h-4 text-black focus:ring-0"
						/>
						<label for="attentionOnly" class="text-sm"> Attention-only (no MLP layers) </label>
					</div>
				</div>
			</div>

			<div class="bg-gray-100 p-4">
				<h2 class="text-lg font-semibold mb-4">Optimizer</h2>

				<div class="form-group mb-4">
					<label class="block text-sm font-medium mb-1">Optimizer Type</label>
					<div class="relative">
						<select
							bind:value={optimizer_type}
							class="w-full p-2 pr-8 border focus:outline-none focus:border-gray-400 border-gray-400 bg-white appearance-none"
						>
							<option value="adamw">AdamW</option>
							<option value="sgd">SGD</option>
						</select>
						<div
							class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400"
						>
							<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path
									d="M19 9l-7 7-7-7"
									stroke-linecap="round"
									stroke-linejoin="round"
									stroke-width="2"
								/>
							</svg>
						</div>
					</div>
				</div>

				<LogSlider bind:value={lr} minExp={-4} maxExp={0} label="Learning Rate" />

				{#if optimizer_type === 'adamw'}
					<button
						class="text-sm text-gray-600 flex items-center gap-2 mt-4"
						on:click={() => (showAdvanced = !showAdvanced)}
					>
						<span class="transform transition-transform" class:rotate-90={showAdvanced}>▶</span>
						Advanced Settings
					</button>

					{#if showAdvanced}
						<div class="mt-4 space-y-4">
							<div class="flex gap-4">
								<div class="form-group flex-1">
									<label class="block text-sm font-medium mb-1">Beta 1</label>
									<input
										type="number"
										min="0"
										max="1"
										step="0.1"
										bind:value={beta1}
										class="w-full p-2 border border-gray-400 focus:outline-none focus:border-gray-400 bg-white"
									/>
								</div>

								<div class="form-group flex-1">
									<label class="block text-sm font-medium mb-1">Beta 2</label>
									<input
										type="number"
										min="0"
										max="1"
										step="0.01"
										bind:value={beta2}
										class="w-full p-2 border border-gray-400 focus:outline-none focus:border-gray-400 bg-white"
									/>
								</div>
							</div>

							<div class="form-group">
								<LogSlider bind:value={eps} minExp={-10} maxExp={-6} label="Epsilon" />
							</div>

							<div class="form-group">
								<LogSlider bind:value={weight_decay} minExp={-4} maxExp={0} label="Weight Decay" />
							</div>
						</div>
					{/if}
				{/if}

				<div class="form-group mt-4">
					<label class="block text-sm font-medium mb-1">Learning Rate Scheduler</label>
					<div class="relative">
						<select
							bind:value={scheduler_type}
							class="w-full p-2 pr-8 border focus:outline-none focus:border-gray-400 border-gray-400 bg-white appearance-none"
						>
							<option value="none">None</option>
							<option value="constant">Constant</option>
							<option value="linear">Linear</option>
							<option value="cosine">Cosine Annealing</option>
						</select>
						<div
							class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400"
						>
							<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path
									d="M19 9l-7 7-7-7"
									stroke-linecap="round"
									stroke-linejoin="round"
									stroke-width="2"
								/>
							</svg>
						</div>
					</div>
				</div>

				{#if scheduler_type !== 'none'}
					<div class="mt-4 space-y-4">
						<div class="form-group">
							<label class="block text-sm font-medium mb-1">Scheduler Steps</label>
							<input
								type="number"
								min="1"
								step="1"
								bind:value={scheduler_steps}
								class="w-full p-2 border border-gray-400 focus:outline-none focus:border-gray-400 bg-white"
							/>
						</div>

						{#if scheduler_type === 'constant' || scheduler_type === 'linear'}
							<div class="form-group">
								<label class="block text-sm font-medium mb-1">Factor</label>
								<input
									type="number"
									min="0"
									max="1"
									step="0.1"
									bind:value={scheduler_factor}
									class="w-full p-2 border border-gray-400 focus:outline-none focus:border-gray-400 bg-white"
								/>
							</div>
						{/if}

						{#if scheduler_type === 'cosine'}
							<div class="form-group">
								<label class="block text-sm font-medium mb-1">Minimum Learning Rate</label>
								<input
									type="number"
									min="0"
									step="0.00001"
									bind:value={scheduler_eta_min}
									class="w-full p-2 border border-gray-400 focus:outline-none focus:border-gray-400 bg-white"
								/>
							</div>
						{/if}
					</div>
				{/if}
			</div>

			<div class="bg-gray-100 p-4">
				<h2 class="text-lg font-semibold mb-4">Training Options</h2>

				<div class="form-group mb-4">
					<TickSlider bind:value={batch_size} min={1} max={10} label="Batch Size" />
				</div>

				<div class="form-group">
					<label class="block text-sm font-medium mb-1">Random Seed (optional)</label>
					<input
						type="text"
						bind:value={seed}
						placeholder="Enter any text for seed"
						class="w-full p-2 border focus:outline-none focus:border-gray-400 border-gray-400 bg-white"
					/>
				</div>
			</div>
		</div>
	</div>
</main>
