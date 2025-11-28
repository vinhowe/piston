<script lang="ts">
	import type { Point, RunData } from '$lib/workspace/runs.svelte';
	import type { LineSeriesOption } from 'echarts/charts';
	import type { CallbackDataParams } from 'echarts/types/dist/shared';

	import createEChartsAttachment, {
		exactIndex,
		extractStepFromAxisPointerEvent,
		nearestIndex,
		publishClear,
		publishMove,
		subscribe
	} from '$lib/attachments/echarts.svelte';
	import { getCurrentRun, getLastNRuns } from '$lib/workspace/runs.svelte';
	import { LineChart } from 'echarts/charts';
	import {
		GridComponent,
		LegendComponent,
		TitleComponent,
		ToolboxComponent,
		TooltipComponent
	} from 'echarts/components';
	import * as echarts from 'echarts/core';
	import { CanvasRenderer } from 'echarts/renderers';
	import { SvelteMap } from 'svelte/reactivity';

	echarts.use([
		TitleComponent,
		TooltipComponent,
		GridComponent,
		LegendComponent,
		ToolboxComponent,
		LineChart,
		CanvasRenderer
	]);

	let { metricName = '' }: { metricName?: string } = $props();

	let containerEl: HTMLDivElement | null = null;
	let isVisible = $state(false);
	const chartId: string = crypto.randomUUID();

	// Mappings for fast lookup during pointer sync
	let seriesIndexToRunId: string[] = [];
	let runIdToSeries: SvelteMap<
		string,
		{ seriesIndex: number; steps: number[]; first: number; last: number }
	> = new SvelteMap();

	$effect(() => {
		if (!containerEl) return;
		const observer = new IntersectionObserver(
			([entry]) => {
				isVisible = entry.isIntersecting;
			},
			{ root: null, threshold: 0.1 }
		);
		observer.observe(containerEl);
		return () => observer.disconnect();
	});

	const chartOptions = $derived.by(() => {
		if (!isVisible) return null;
		const series: LineSeriesOption[] = [];
		const legendData: string[] = [];
		const currentRuns = getLastNRuns(5);
		const currentRunId = getCurrentRun()?.runId;

		// Reset mappings
		seriesIndexToRunId = [];
		runIdToSeries = new SvelteMap();

		// Reverse the order so most recent runs are drawn on top (last in series array)
		currentRuns
			.slice()
			.reverse()
			.forEach((run: RunData) => {
				const metric = run.metrics.get(metricName);
				if (metric) {
					const seriesName = `${run.runId}`;
					legendData.push(seriesName);
					const chartData = metric.data
						.filter((p): p is Point => 'y' in p)
						.map((p) => [p.step, p.y]);
					const isCurrentRun = run.runId === currentRunId;
					const opacity = isCurrentRun ? 1.0 : 0.4;

					series.push({
						id: seriesName,
						name: seriesName,
						type: 'line',
						color: run.color,
						sampling: 'minmax',
						lineStyle: {
							width: 2,
							opacity: opacity
						},
						itemStyle: {
							opacity: opacity
						},
						data: chartData,
						showSymbol: false,
						emphasis: {
							focus: 'series',
							lineStyle: {
								width: 3.5,
								opacity: 1.0
							},
							itemStyle: {
								opacity: 1,
								color: run.color
							}
						},
						symbolSize: 6,
						triggerLineEvent: true
					});

					// Update maps for this series
					const idx = series.length - 1;
					seriesIndexToRunId[idx] = run.runId;
					const steps = chartData
						.map((d) => {
							if (Array.isArray(d)) return Number(d[0]);
							if (d && typeof d === 'object' && 'x' in (d as Record<string, unknown>)) {
								const raw = (d as Record<string, unknown>).x;
								const n =
									typeof raw === 'number' ? raw : typeof raw === 'string' ? Number(raw) : NaN;
								return Number.isFinite(n) ? n : NaN;
							}
							return NaN;
						})
						.filter((n) => Number.isFinite(n));
					if (steps.length > 0) {
						const first = steps[0];
						const last = steps[steps.length - 1];
						runIdToSeries.set(run.runId, { seriesIndex: idx, steps, first, last });
					}
				}
			});

		const options = {
			animation: false,
			title: {
				text: metricName,
				left: 'center',
				textStyle: {
					fontSize: 13
				},
				itemGap: 0,
				top: 12
			},
			border: 'none',
			backgroundColor: 'transparent',
			tooltip: {
				animation: false,
				transitionDuration: 0,
				trigger: 'axis',
				borderColor: '#ccc',
				borderWidth: 1,
				borderRadius: 0,
				textStyle: {
					color: 'black',
					fontSize: 11
				},
				padding: [6, 8],
				formatter: function (params: CallbackDataParams[]) {
					if (!params || !params.length) {
						return '';
					}

					// Use value[0] for x-axis value, assuming value axis type
					const xAxisValue = Array.isArray(params[0].value) ? params[0].value[0] : params[0].name; // Fallback to name for category axis
					let result = `<div class="leading-none flex flex-col gap-1"><div class="font-mono font-bold">Step ${xAxisValue}</div>`;

					// Sort series by value for better readability - Ensure value is an array and has a second element
					params.sort((a, b) => {
						// Provide numeric fallbacks (e.g., -Infinity) for safe comparison
						const valA =
							Array.isArray(a.value) && a.value.length > 1 && typeof a.value[1] === 'number'
								? a.value[1]
								: -Infinity;
						const valB =
							Array.isArray(b.value) && b.value.length > 1 && typeof b.value[1] === 'number'
								? b.value[1]
								: -Infinity;

						// Handle cases where one or both might be -Infinity
						if (valB === -Infinity && valA === -Infinity) return 0;
						if (valB === -Infinity) return -1; // Put Infinity/valid numbers first
						if (valA === -Infinity) return 1;

						return valB - valA; // Now safe to subtract
					});

					result += '<table style="width: 100%; border-collapse: collapse;">';

					params.forEach((param) => {
						const value =
							Array.isArray(param.value) && param.value.length > 1 ? param.value[1] : undefined;
						const formattedValue = typeof value === 'number' ? value.toFixed(4) : 'N/A';

						result += `
							<tr style="color: ${param.color};">
								<td class="font-mono" style="padding: 2px 4px 2px 0;">
									<span style="display: inline-block; width: 10px; height: 3px; background-color: ${param.color}; margin-right: 2px; margin-bottom: 2px;"></span>
									${formattedValue}
								</td>
								<td class="text-left text-[12px]">
									${param.seriesName}
								</td>
							</tr>
						`;
					});

					result += '</table></div>';
					return result;
				},
				axisPointer: {
					animation: false,
					label: {
						backgroundColor: '#6a7985'
					},
					lineStyle: {
						color: '#555',
						width: 1,
						type: 'solid'
					}
				}
			},
			grid: {
				left: '3%',
				right: '4%',
				bottom: '10%',
				top: '15%',
				containLabel: true
			},
			xAxis: {
				type: 'value',
				axisPointer: {
					snap: true
				}
			},
			yAxis: {
				type: 'value',
				scale: true,
				...((metricName.endsWith('loss') ||
					metricName.endsWith('perplexity') ||
					metricName.endsWith('accuracy')) && { min: 0 }),
				...(metricName.endsWith('accuracy') && { max: 1 })
			},
			series: series
		};
		return options;
	});
</script>

<div class="w-full h-[250px] min-h-[150px] relative" bind:this={containerEl}>
	<div
		class="w-full h-full"
		{@attach createEChartsAttachment({
			opts: () => chartOptions,
			setup: (chart) => {
				let isRelaying = false;

				const onUpdateAxisPointer = (evt: unknown) => {
					if (isRelaying) return;
					const s = extractStepFromAxisPointerEvent(evt);
					if (s == null) return;
					// Choose topmost series that contains this exact step
					for (let idx = seriesIndexToRunId.length - 1; idx >= 0; idx--) {
						const runId = seriesIndexToRunId[idx];
						if (!runId) continue;
						const seriesInfo = runIdToSeries.get(runId);
						if (!seriesInfo) continue;
						if (exactIndex(seriesInfo.steps, s) !== -1) {
							publishMove({ sourceId: chartId, runId, step: s });
							break;
						}
					}
				};

				const onGlobalOut = () => {
					publishClear({ sourceId: chartId });
				};

				chart.on('updateAxisPointer', onUpdateAxisPointer);
				chart.on('globalout', onGlobalOut);

				const unsubscribe = subscribe(
					({ sourceId, runId, step }) => {
						if (sourceId === chartId) return;
						const info = runIdToSeries.get(runId);
						if (!info) return;
						const { seriesIndex, steps, first, last } = info;
						if (step < first || step > last) return;
						const dataIndex = nearestIndex(steps, step);
						if (dataIndex < 0) return;
						isRelaying = true;
						try {
							// ECharts accepts a second options argument with { silent: true }
							chart.dispatchAction(
								{ type: 'updateAxisPointer', seriesIndex, dataIndex },
								{ silent: true }
							);
						} finally {
							isRelaying = false;
						}
					},
					({ sourceId }) => {
						if (sourceId === chartId) return;
						chart.dispatchAction({ type: 'hideTip' }, { silent: true });
					}
				);

				return () => {
					unsubscribe();
					chart.off('updateAxisPointer', onUpdateAxisPointer);
					chart.off('globalout', onGlobalOut);
				};
			}
		})}
	></div>
</div>
